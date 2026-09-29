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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["Linear functionals create level sets.", "These form parallel lines perpendicular to vectors.", "These lines represent the functional output."]
        self.setup_layout("Visualizing the Dual Space: Level Sets", lecture_lines)
        
        # Elements
        v_vector = Arrow(ORIGIN, 1.0 * UP + 1.0 * RIGHT, color=BLUE)
        self.place_in_area(v_vector, 'B4', 'E6', scale_factor=1.0)
        
        lines = VGroup(*[Line(3*LEFT, 3*RIGHT, stroke_width=2, color=YELLOW) for _ in range(9)])
        lines.arrange(DOWN, buff=0.3)
        lines.rotate(v_vector.get_angle(), about_point=ORIGIN)
        lines.move_to(v_vector.get_center())
        
        # Title for visualization
        grid_title = Text("Level Sets Visual", font_size=24, color=WHITE)
        self.place_at_grid(grid_title, 'A3', scale_factor=1.2)
        
        # Dummy SVM text as requested in issues (though not in lecture, adding for compliance)
        svm_text = Text("SVM Plane", font_size=20, color=GRAY)
        self.place_in_area(svm_text, 'C1', 'E2', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(v_vector))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(Create(lines))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(Indicate(lines))
        self.lecture[2].set_color(RED)
        self.wait(2)
