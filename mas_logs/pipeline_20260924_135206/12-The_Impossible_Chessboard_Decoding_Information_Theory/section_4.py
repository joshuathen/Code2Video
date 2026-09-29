from manim import *

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Flipping one coin changes exactly one parity bit.",
            "This effectively corrects the total parity sum.",
            "Parity acts as an error detection mechanism.",
            "Mathematically map board state to unique square.",
            "Proof confirmed: always identify the secret square."
        ]
        self.setup_layout("Mathematical Proof via Parity", lecture_lines)
        
        # Setup visuals: Input bits (4x4 example)
        bits = VGroup(*[Square(side_length=0.5, color=WHITE).add(Text(str(i % 2), font_size=20)) for i in range(16)])
        bits.arrange_in_grid(4, 4, buff=0.1)
        self.place_in_area(bits, 'A2', 'C4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(FadeIn(bits))
        
        # Asset: coin.svg
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        self.place_at_grid(coin, 'B2', scale_factor=0.5)
        self.play(Flash(coin, color="#00FF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        xor_rect = Rectangle(width=2.2, height=2.2, color="#FFFF00").move_to(bits.get_center())
        self.play(Create(xor_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        result_bit = Text("Parity Output", font_size=20, color="#00FF00")
        self.place_at_grid(result_bit, 'D3', scale_factor=0.7)
        self.play(Write(result_bit))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FFFF")
        mapping_text = Text("Mapping: State -> Square", font_size=20, color="#00FFFF")
        self.place_at_grid(mapping_text, 'E3', scale_factor=0.8)
        self.play(FadeIn(mapping_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        confirm_text = Text("Proven!", font_size=24, color="#FF00FF")
        self.place_at_grid(confirm_text, 'E4', scale_factor=1.0)
        self.play(Flash(confirm_text))
        self.wait(2)
