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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Circles exist in every size.", "A coin is a circle.", "A plate is a circle.", "A Ferris wheel is a circle.", "All share a special constant ratio."]
        self.setup_layout("The Hook: The Constant Ratio", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        circle = Circle(radius=0.8, color=WHITE)
        diameter = Line(circle.get_left(), circle.get_right(), color=WHITE)
        plate_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg", color=WHITE)
        self.place_at_grid(VGroup(circle, diameter), 'B2', scale_factor=0.8)
        self.play(Create(circle), Create(diameter))
        self.play(Transform(circle, plate_asset.move_to(circle.get_center())))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        coin = Circle(radius=0.4, color=YELLOW)
        coin_label = Text("Coin", font_size=20, color=YELLOW)
        self.place_at_grid(coin, 'D1', scale_factor=0.7)
        self.place_at_grid(coin_label, 'E1', scale_factor=0.5)
        self.play(FadeIn(coin), Write(coin_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        plate = Circle(radius=0.5, color=YELLOW)
        plate_label = Text("Plate", font_size=20, color=YELLOW)
        self.place_at_grid(plate, 'D3', scale_factor=0.7)
        self.place_at_grid(plate_label, 'E3', scale_factor=0.5)
        self.play(FadeIn(plate), Write(plate_label))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        wheel = Circle(radius=0.6, color=YELLOW)
        wheel_label = Text("Wheel", font_size=20, color=YELLOW)
        self.place_at_grid(wheel, 'D5', scale_factor=0.7)
        self.place_at_grid(wheel_label, 'E5', scale_factor=0.5)
        self.play(FadeIn(wheel), Write(wheel_label))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        ratio_text = MathTex(r"C/D = \text{constant}", color=PURPLE)
        coin_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color=YELLOW).scale(0.3)
        self.place_in_area(ratio_text, 'B3', 'C5', scale_factor=1.0)
        self.play(Write(ratio_text))
        self.play(FadeIn(coin_asset.next_to(ratio_text, RIGHT)))
        self.wait(2)
