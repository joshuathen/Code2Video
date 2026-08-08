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
        self.setup_layout("Understanding Discrete Convolutions", ["A filter slides across your image data.", "It extracts patterns by looking at local areas.", "Think of it like a smart focus lens."])
        
        # Grid Setup
        input_data = VGroup(*[Square(side_length=0.7, color=WHITE, fill_opacity=0.3) for _ in range(16)])
        input_data.arrange_in_grid(4, 4, buff=0.05)
        self.place_in_area(input_data, "C3", "F6", scale_factor=0.55)
        
        # Load asset
        lens = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/lens.svg", color="#FF5733")
        self.place_at_grid(lens, "A6", scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF5733"), run_time=0.5)
        self.add(lens)
        self.play(
            lens.animate.move_to(self.grid["B3"]),
            run_time=1.0
        )
        self.play(
            lens.animate.move_to(self.grid["B4"]),
            run_time=1.0
        )

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"), run_time=0.5)
        # Highlight logic
        highlight = SurroundingRectangle(input_data[0:9], color="#FFFF00", buff=0.05)
        self.play(Create(highlight), run_time=1.0)
        self.play(FadeOut(highlight), run_time=1.0)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), run_time=0.5)
        self.play(
            lens.animate.move_to(self.grid["C3"]),
            run_time=1.5
        )
        self.wait(1)
