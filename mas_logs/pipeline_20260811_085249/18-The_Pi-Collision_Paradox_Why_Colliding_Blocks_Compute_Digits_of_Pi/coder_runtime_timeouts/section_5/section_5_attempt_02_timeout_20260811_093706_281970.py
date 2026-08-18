from manim import *
import os

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: Computation through Nature", ["Nature computes complex math.", "Two blocks act as calculators.", "Pi is hidden in motion."])
        
        # Asset path
        chip_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/chip.svg"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        # Show initial machine (simplified collision)
        wall = Line(start=UP*1.5, end=DOWN*1.5, color=GREY).shift(RIGHT*3)
        block1 = Square(side_length=0.5, color=BLUE, fill_opacity=0.8)
        block2 = Square(side_length=0.8, color=RED, fill_opacity=0.8)
        calculator_group = VGroup(wall, block1, block2)
        
        # Applying requested fix for calculator position
        self.place_in_area(calculator_group, 'B4', 'E6', scale_factor=0.9)
        self.play(Create(calculator_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        # Load and morph to chip
        if os.path.exists(chip_path):
            chip = SVGMobject(chip_path)
        else:
            chip = RoundedRectangle(corner_radius=0.2, height=2, width=3, color="#00FF00")
            
        chip.set_color("#00FF00")
        self.place_at_grid(chip, 'D3', scale_factor=1.5)
        
        # Grid visual improvement
        grid_lines = VGroup()
        for i in range(7):
            grid_lines.add(Line(start=LEFT*2+UP*(2.5-i*1), end=RIGHT*2+UP*(2.5-i*1), color=GRAY))
            grid_lines.add(Line(start=UP*2.5+LEFT*(2-i*0.8), end=DOWN*2.5+LEFT*(2-i*0.8), color=GRAY))
            
        self.place_in_area(grid_lines, 'A2', 'F6', scale_factor=0.85)
        
        self.play(
            FadeOut(calculator_group),
            FadeIn(grid_lines),
            ReplacementTransform(calculator_group, chip)
        )

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        # Flash result 3.14
        result_text = Text("3.14", font_size=72, color=WHITE)
        self.place_at_grid(result_text, 'D5', scale_factor=1.0)
        
        self.play(Write(result_text))
        self.play(result_text.animate.set_color(YELLOW), run_time=0.5)
        self.play(result_text.animate.set_color(WHITE), run_time=0.5)
        self.play(FadeOut(result_text, chip, grid_lines))
