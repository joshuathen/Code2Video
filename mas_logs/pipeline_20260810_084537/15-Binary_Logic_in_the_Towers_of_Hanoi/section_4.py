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
        lecture_lines = ["A robot solves it by counting binary.", "The counter triggers mechanical arm moves.", "Automation removes the need for strategy."]
        self.setup_layout("Application: The Automated Robot Strategy", lecture_lines)
        
        # Elements
        robot_head = Circle(radius=0.5, color=BLUE).set_fill(BLUE, opacity=0.5)
        binary_display = Text("000", font_size=36, color=YELLOW)
        arm = Line(ORIGIN, RIGHT * 1.5, color=WHITE, stroke_width=8)
        
        # === Animation for Lecture Line 1 ===
        # A robot solves it by counting binary.
        self.place_at_grid(robot_head, 'C3', scale_factor=0.8)
        self.place_at_grid(binary_display, 'C4', scale_factor=0.7)
        self.play(FadeIn(robot_head), FadeIn(binary_display))
        self.lecture[0].set_color(BLUE)
        
        # === Animation for Lecture Line 2 ===
        # The counter triggers mechanical arm moves.
        self.place_at_grid(arm, 'E4', scale_factor=0.9)
        self.play(Create(arm), binary_display.animate.set_text("001"))
        self.play(arm.animate.rotate(PI/4, about_point=self.grid['E4']))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Automation removes the need for strategy.
        check_mark = Tex(r"$\checkmark$", color=GREEN).scale(2)
        self.place_at_grid(check_mark, 'E5', scale_factor=0.6)
        self.play(FadeIn(check_mark))
        self.lecture[2].set_color(GREEN)
        self.wait(2)
