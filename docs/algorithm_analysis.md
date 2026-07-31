# Phan tich thuat toan CF-SCD (Coverage Fingerprint - Spatial Change Detection)

Tai lieu nay phan tich thuat toan CF-SCD va doi chieu voi phan hien thuc
trong `coverage_intelligence/`.

## 1. Ban chat thuat toan

CF-SCD giai mot bai toan cu the: **phat hien bien dong vung phu cua tung
cell theo khong gian - thoi gian, som hon va chi tiet hon KPI tong hop**, roi
**giai thich nguyen nhan** va **luong hoa** ket qua thanh mot chi so quan
tri thong nhat (Coverage Health Score).

Diem khac biet cot loi so voi giam sat KPI truyen thong: thay vi rut gon
trang thai vung phu ve mot vai gia tri trung binh (RSRP trung binh,
Accessibility, Retainability...), CF-SCD giu lai **hinh thai phan bo** cua
tin hieu theo khoang cach va huong (Coverage Fingerprint), nen phat hien
duoc cac suy giam cuc bo bi "bu tru" trong gia tri trung binh toan cell -
day chinh la luan de khoa hoc cot loi cua thuat toan.

## 2. Bay buoc xu ly va anh xa sang code

| Buoc thuat toan | Module hien thuc | Ghi chu ky thuat |
|---|---|---|
| 1. Trich xuat dac trung vung phu | `features.py` | 5 nhom dac trung: Signal, Distance, Ring, Direction, Grid |
| 2. Xay dung Coverage Fingerprint | `features.compute_fingerprints` | Vector hoa moi (cell, ngay) thanh mot hang; rieng Ring x Direction duoc giu la ma tran 2 chieu de so sanh "hinh dang" |
| 3. Xay dung duong chuan (Baseline) | `baseline.py` | Median + MAD tren cua so 7-30 ngay, loai ngay ngoai lai/su co da xac nhan; co them baseline theo nhom peer (cung band/tech) cho cell moi |
| 4. Phat hien bien dong khong gian | `detection.py` | Rule-based + Spatial Similarity Index (SSI) + EWMA/CUSUM/STL-lite, hop nhat bang Ensemble Voting |
| 5. Phan tich nguyen nhan & de xuat | `rca.py` | Tuong quan cell lan can + thu vien luat chuyen gia + doi chieu Alarm |
| 6. Danh gia Coverage Health Score | `health_score.py` | 5 thanh phan co trong so, phan hang Excellent/Good/Warning/Critical, tong hop theo phan cap |
| 7. Hoc hoi va cai tien lien tuc | `pipeline.FeedbackStore` | Ghi nhan xac nhan cua ky su -> loai ngay su co that khoi baseline lan sau |

## 3. Danh gia tung buoc

### 3.1. Trich xuat dac trung & Coverage Fingerprint (buoc 1-2)

Diem manh: nam nhom dac trung (Signal/Distance/Ring/Direction/Grid) bao phu
ca chieu muc tin hieu lan chieu khong gian, phu hop de phat hien bien dong
ma khong can toa do UE chinh xac tuyet doi (chi can khoang cach + phuong vi
tinh tu vi tri tram).

Rui ro can luu y khi trien khai thuc te: **do phan giai Ring x Direction
phai can bang voi mat do mau**. Trong qua trinh xay dung ban demo, voi
~150 mau/cell/ngay chia cho 5 vong x 12 huong = 60 o, phan lon o gan nhu
rong -> chi so tuong tu khong gian (SSI) mat kha nang phan biet. Khi tang
len ~1500 mau/cell/ngay (van thap hon nhieu so voi luu luong UE Report thuc
te cua mot mang di dong quy mo lon), SSI moi on dinh. Day la mot rang buoc
thiet ke can nhac khi chon do phan giai luoi: **so o (rings x directions x
grid) phai duoc dat theo mat do mau thuc te cua tung loai cell** (macro/nong
thon it mau hon micro/do thi), khong nen dung mot do phan giai co dinh cho
toan mang.

### 3.2. Xay dung Baseline (buoc 3)

MAD duoc chon thay vi do lech chuan la hop ly cho du lieu vien thong (nhieu
gia tri ngoai lai do bien dong luu luong). Diem can luu y khi hien thuc:
**"loai ngoai lai" va "loai ngay su co da xac nhan" nen dung chung mot bo
tieu chi cho toan bo cac chieu dac trung**, thay vi xu ly rieng le tung
chieu - neu khong, mot ngay co the "binh thuong" o dac trung nay nhung "bat
thuong" o dac trung khac, lam baseline khong nhat quan. Ban hien thuc chon
loai ngay dua tren mot chi so tong hop (RSRP trung binh) roi ap dung dong
loat cho toan bo vector fingerprint.

Truong hop cell moi (chua du 7-30 ngay lich su) duoc kich ban Overshoot nhac
toi ("cho phep danh gia chinh xac ngay ca voi tram moi") - day la mot yeu
cau ro rang can co **co che baseline thay the** (so sanh cheo voi nhom peer
cung band/tech/cong nghe tai cung thoi diem) thay vi chi dung lich su rieng
cua cell, va ban hien thuc trien khai co che nay (`baseline.peer_group_baseline`).

### 3.3. Phat hien bien dong (buoc 4)

Day la buoc quan trong nhat va cung la noi de sai lech nhat neu chi thiet ke
tren giay ma khong thu nghiem so lieu:

- **Rule-based** de hieu, de giai thich voi ky su van hanh, nhung chi bat
  duoc bien dong bien do lon.
- **SSI** la thanh phan bat duoc dung luan de khoa hoc cot loi cua thuat
  toan (bien dong cuc bo khong the hien qua gia tri trung binh) - nhung
  **nguong canh bao cua SSI phai duoc hieu chuan tren du lieu thuc te**,
  khong the dat mot con so co dinh nhu 0.8 ma khong kiem tra phan phoi thuc
  te cua chi so nay. Trong thu nghiem tren du lieu mo phong, SSI "binh
  thuong" (khong co bien dong) dao dong quanh 0.99 va chi giam xuong
  0.94-0.97 voi cac bien dong cuc bo tinh vi (che chan, lech huong) - nghia
  la **do nhay thuc te cua SSI rat cao (phai phan biet trong khoang
  0.94-0.999) chu khong phai 0-1 nhu ve mat ly thuyet**. Day la diem ban
  thiet ke ban dau khong the hien ro va can duoc luu y khi trien
  khai/hieu chuan that.
- **Time-series (EWMA/CUSUM/STL)** phu hop voi suy giam cham - trong ban
  hien thuc, STL "day du" (mua vu + xu huong + phan du qua statsmodels) duoc
  thay bang mot phien ban rut gon (do lech so voi trung binh truot ngan
  han) vi chuoi ngay quan sat trong demo qua ngan de tach mua vu co y nghia;
  voi du lieu that (nhieu thang), nen dung STL day du.
- **Ensemble Voting** (>=2/3 phuong phap dong thuan) la lua chon dung dan
  de giam canh bao gia - thu nghiem cho thay day la co che quyet dinh de
  giu ty le canh bao gia gan 0% tren cac cell khong co bien dong (sau giai
  doan "khoi dong" baseline).

### 3.4. Phan tich nguyen nhan (buoc 5)

Thiet ke mo ta dung huong nhung o muc kha tong quat ("doi chieu thu vien
luat chuyen gia"). Trong qua trinh hien thuc, diem mau chot de phan biet
chinh xac giua cac nguyen nhan la **tim ra tin hieu dac trung rieng cho
tung loai**, khong chi dua vao muc do bien dong tong quat:

- *Che chan (Shadowing)* vs *Lech Azimuth*: ca hai deu lam SSI giam va lam
  huong "dinh" tin hieu dich chuyen, nhung che chan **chi lam mat tin hieu**
  (khong co huong nao duoc cai thien), trong khi lech Azimuth **phan phoi
  lai** tin hieu (mot so huong xau di, mot so huong tot len vi bup song
  huong sang do). Chi so "ty le sector duoc cai thien" (frac_improved) la
  yeu to phan biet manh nhat trong thu nghiem.
- *Sai lech Tilt/cong suat* vs *Overshoot*: hai kich ban nay doi ngau nhau
  ve dau cua bien dong ban kinh hieu dung (dist_p90) - Tilt/cong suat lam
  ban kinh **co lai** kem RSRP giam, Overshoot lam ban kinh **mo rong**
  vuot chuan (so voi baseline rieng hoac nhom peer).

Diem can luu y khi trien khai that: **khong nen dua vao mot "ban kinh thiet
ke" tuyet doi de xac dinh Overshoot** (vi day thuong khong phai la mot con
so co san, chinh xac trong CSDL cau hinh) ma nen so sanh voi baseline lich
su hoac nhom peer - dung nhu thiet ke ban dau va dung nhu ban hien thuc ap
dung sau khi sua loi ban dau (phien ban dau cua code demo dung truc tiep
"design_radius" tong hop, dan den sai lech RCA co he thong - da duoc sua lai
de chi dung baseline/peer, phan anh dung tinh than cua thiet ke ban dau).

### 3.5. Coverage Health Score (buoc 6)

Cong thuc goc chi neu cac thanh phan dinh tinh (chat luong tin hieu, on
dinh, muc bien dong, nguyen nhan, anh huong) ma khong cho trong so cu the -
day la lua chon hop ly o giai doan thiet ke ban dau (trong so can hieu
chuan theo du lieu that), nhung cung co nghia **bat ky bo trong so nao dua
ra o giai doan trien khai deu can duoc kiem chung bang du lieu Ground Truth
thuc te** (vi du: CHS co tuong quan voi so luot phan anh cua khach hang
hay khong), khong chi dua vao truc giac ky thuat.

### 3.6. Hoc hoi lien tuc (buoc 7)

Day la co che dam bao thuat toan khong "dong cung" theo thoi gian. Diem
quan trong can lam ro khi trien khai: phan hoi cua ky su nen tac dong o
**hai cap do khac nhau** - (a) loai ngay su co that khoi baseline (tranh
"tu lam ban than" - anh huong da duoc xac nhan khong nen tro thanh "binh
thuong moi"), va (b) dieu chinh trong so/nguong cua thu vien luat RCA. Ban
hien thuc (`pipeline.FeedbackStore`) moi giai quyet (a) mot cach tu dong;
(b) duoc de ngo nhu mot diem mo rong co chu dich (chinh sua bang
`rca.RECOMMENDATIONS`/`rca.ALARM_BOOST` bang tay, tien toi mo hinh hoc giam
sat nhu dinh huong dai han ban dau).

## 4. Tong ket

Thuat toan CF-SCD la mot thiet ke hop ly va co co so khoa hoc (phan tich du
lieu khong gian + thong ke ben vung + ensemble + human-in-the-loop), phu
hop de trien khai tung buoc tren ha tang du lieu UE Report/Mentor san co
cua mot mang vo tuyen di dong. Cac diem can bo sung khi chuyen tu thiet ke
sang trien khai thuc te - va da duoc xu ly trong ban hien thuc kem theo -
la: (1) hieu chuan do phan giai khong gian theo mat do mau thuc te, (2)
hieu chuan nguong SSI/EWMA/CUSUM tren phan phoi du lieu that thay vi dat
theo truc giac, (3) dung tin hieu **dau** cua bien dong (khong chi do lon)
de phan biet cac nguyen nhan gan giong nhau, va (4) tranh dua cac "hang so
thiet ke" khong chac chan vao logic phat hien/phan tich - luon uu tien so
sanh voi baseline hoac nhom peer.
