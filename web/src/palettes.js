/** Editorial example palettes, not diagnostic standards. Spring Bright retains v1 references. */
const make = (id, name, family, description, base, accent) => ({ id, name, family, description,
  colors: [base, accent].flatMap((items, i) => items.split('|').map(item => {
    const [label, hex] = item.split(':'); return { name: label, hex: `#${hex}`, role: i ? 'accent' : 'base' };
  })) });
export const PALETTES = [
  make('spring-bright', '봄 브라이트', '봄', '따뜻하고 선명한 색을 중심으로 탐색해 봐.',
    '아이보리:FFF7E6|크림:FDF0D5|라이트 베이지:EED9B6|버터 옐로:FFE48A|카멜:C98A3F|네이비:103A64',
    '코랄:FF6B5C|토마토 레드:FF3B30|피치:FFA07A|망고:FFC107|애플 그린:5CD85C|민트:58E0C0|아쿠아:6DCFF6|티얼:0BB3A6'),
  make('spring-light', '봄 라이트', '봄', '가벼운 크림색과 맑은 파스텔을 살펴봐.',
    '크림:FFF5E1|오트밀:EADBC4|라이트 카멜:C4A476|소프트 네이비:526D82',
    '살구:FFBD9B|라이트 코랄:FF9990|버터:FFE995|라이트 민트:A9E8CA|하늘:AFE2F3|멜론:DFEC9C'),
  make('spring-warm', '봄 웜', '봄', '노랑빛이 느껴지는 따뜻하고 생생한 색의 예시야.',
    '웜 아이보리:FFF0D2|카멜:BC873C|웜 브라운:825230|올리브:808F3D',
    '오렌지:FF9146|토마토:EE5940|골든 옐로:F6C544|리프 그린:89B34A|터쿼이즈:38BCA7|피치:FFAD7E'),
  make('summer-light', '여름 라이트', '여름', '밝고 시원한 파스텔로 부드러운 조합을 만들어 봐.',
    '펄 화이트:F5F5F4|쿨 베이지:DCD6D1|실버 그레이:BCC5CF|블루 그레이:74889C',
    '페일 핑크:F2BCCD|라벤더:CEC1EB|스카이:B9DEF0|파우더 민트:BFE3DA|로즈:E6A5BC|라일락:DDC1E0'),
  make('summer-cool', '여름 쿨', '여름', '차분한 블루와 로즈 계열을 비교해 봐.',
    '소프트 화이트:F0F1F5|쿨 그레이:B0B8C4|네이비:384D71|코코아:897780',
    '로즈:D7789A|베리:B85988|블루:719AC9|라벤더:A58FC4|페리윙클:929CCE|아쿠아:79BCC5'),
  make('summer-soft', '여름 소프트', '여름', '회색기가 섞인 은은한 색의 조합이야.',
    '오이스터:E7E2DB|로즈 베이지:C7B6AE|토프:978E8B|차콜:5B626D',
    '더스티 로즈:BF909C|모브:AB8BA5|세이지:A2B6AA|스모키 블루:87A5BA|라벤더:ACA1BF|로즈우드:9D6476'),
  make('autumn-soft', '가을 소프트', '가을', '따뜻한 흙빛과 부드러운 초록을 살펴봐.',
    '에크루:EDE0C8|샌드:CDB591|토프:A39480|코코아:7D6855',
    '살몬:D9947D|세이지:A4AD87|올리브:8D9663|더스티 피치:D8AA89|뮤트 티얼:73978D|클레이:B9816D'),
  make('autumn-warm', '가을 웜', '가을', '황금빛과 테라코타를 중심으로 묶은 예시야.',
    '크림:EEDDAD|카멜:B98542|초콜릿:6E422B|모스:727342',
    '테라코타:BF6545|머스터드:C89E36|번트 오렌지:CC7534|올리브:8E9139|페트롤:347F77|러스트:A44B32'),
  make('autumn-deep', '가을 딥', '가을', '깊은 브라운과 진한 보석빛을 함께 비교해 봐.',
    '아이보리:EEDFC2|에스프레소:3E2B25|다크 올리브:4D5233|웜 네이비:273E43',
    '옥스블러드:713A35|딥 티얼:176C61|포레스트:355C3A|러스트:A24B31|골드:B38A39|오버진:60414A'),
  make('winter-deep', '겨울 딥', '겨울', '짙은 바탕에 또렷한 포인트를 놓아 봐.',
    '쿨 화이트:F3F4F7|차콜:343640|블랙:17171F|잉크 네이비:202B48',
    '버건디:7D2447|에메랄드:006653|사파이어:274B9A|딥 퍼플:52376F|크랜베리:B9224F|아이스 핑크:E8CBDF'),
  make('winter-cool', '겨울 쿨', '겨울', '시원하고 또렷한 블루·핑크 계열의 예시야.',
    '화이트:FAFAFF|실버:C3C7D1|쿨 차콜:434855|네이비:27375C',
    '블루 레드:CE2553|마젠타:C92C8B|코발트:355EC6|바이올렛:7051AC|쿨 그린:008E7B|아이스 블루:CFE9F4'),
  make('winter-bright', '겨울 브라이트', '겨울', '강한 대비와 맑은 포인트로 실험해 봐.',
    '화이트:FFFFFF|블랙:171B26|실버:D2D6DF|미드나잇:1C2A54',
    '푸시아:ED3293|일렉트릭 블루:2359E0|체리:F03558|에메랄드:00A978|아이스 민트:D0F1EC|레몬:F5EF65'),
];
export const getPalette = id => PALETTES.find(p => p.id === id) || null;
export const DEFAULT_COLOR = '#FF6B5C';
export const ALL_COLORS = [...new Map(PALETTES.flatMap(p => p.colors).map(c => [c.hex, c])).values()];
