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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Criterion 1: Visual-Conceptual Coupling", [
            "Anchor abstract symbols to visual maps.",
            "Graphs represent geometric laws of algebra.",
            "See the derivative as a changing slope."
        ])
        
        # Assets
        light_icon_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFCCCC")
        input_block = Rectangle(width=2.5, height=1.5, color="#FFFFFF").set_fill(opacity=0.3)
        input_label = Text("Input", font_size=24, color=WHITE)
        light_icon = SVGMobject(light_icon_path).set_color(WHITE)
        input_group = VGroup(input_block, input_label, light_icon).arrange(DOWN, buff=0.2)
        self.place_at_grid(input_group, 'B3', scale_factor=0.8)
        self.play(Create(input_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFCC")
        beam = Line(start=input_group.get_right(), end=self.grid['B4'], color="#FFFF00", stroke_width=6)
        self.play(Create(beam))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#CCFFFF")
        output_text = Text("Visual Output", font_size=24, color="#00FFFF")
        light_icon_2 = SVGMobject(light_icon_path).set_color(WHITE)
        output_group = VGroup(output_text, light_icon_2).arrange(DOWN, buff=0.2)
        self.place_at_grid(output_group, 'B4', scale_factor=0.8)
        self.play(Transform(beam, output_group))
        self.wait(2)
