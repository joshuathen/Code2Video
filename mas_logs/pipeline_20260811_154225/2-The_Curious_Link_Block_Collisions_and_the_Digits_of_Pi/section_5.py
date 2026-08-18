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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Nature acts as a mechanical Pi calculator.",
            "Physical motion maps to abstract mathematical constants.",
            "We see Pi through these mechanical collisions."
        ]
        self.setup_layout("Conclusion: Computation through Nature", lecture_lines)
        
        # Prep assets
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg]
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        calc_icon.set_color("#00FF00")
        calc_label = Text("Calculator", font_size=20)
        
        system_icon = Circle(radius=0.75, color="#00FF00")
        system_label = Text("Nature", font_size=20)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(calc_icon, "B2", scale_factor=0.7)
        self.place_at_grid(calc_label, "C2", scale_factor=0.8)
        self.play(Create(calc_icon), Write(calc_label))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_at_grid(system_icon, "B5", scale_factor=0.7)
        self.place_at_grid(system_label, "C5", scale_factor=0.8)
        self.play(Create(system_icon), Write(system_label))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        final_text = Text("Nature Computes", font_size=36, color="#FFFFFF")
        self.place_in_area(final_text, "D2", "E5", scale_factor=0.9)
        self.play(Write(final_text), Flash(final_text, color="#FFFFFF", line_length=0.2, num_lines=15))
        self.wait(2)
