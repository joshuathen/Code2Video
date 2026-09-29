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
        self.setup_layout("Summary & Quick Check", [
            "Always treat y as a function.",
            "Implicit methods handle non-function shapes.",
            "Derivative of y^4 is 4y^3*dy/dx."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Flash summary of key procedural steps.
        step1 = Text("Differentiate", color=YELLOW)
        step2 = Text("Collect", color=YELLOW)
        step3 = Text("Solve", color=YELLOW)
        steps = VGroup(step1, step2, step3).arrange(DOWN, aligned_edge=LEFT)
        self.place_at_grid(steps, 'B5', scale_factor=0.6)
        self.play(FadeIn(steps), Indicate(steps))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        # Highlight 'differentiate, collect, solve' text.
        self.play(steps.animate.set_color(BLUE), run_time=1)
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        # Display a final 'Q&A' icon briefly.
        qa_label = Text("Q&A", font_size=48)
        qa_box = SurroundingRectangle(qa_label, color=WHITE, buff=0.3)
        qa_group = VGroup(qa_label, qa_box)
        self.place_in_area(qa_group, 'D5', 'F6', scale_factor=0.8)
        self.play(Write(qa_group))
        self.play(self.lecture[2].animate.set_color(RED))
        self.wait(2)
