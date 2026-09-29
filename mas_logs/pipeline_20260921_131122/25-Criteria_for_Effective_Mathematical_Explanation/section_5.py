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
        self.setup_layout("Conclusion: The Feedback Loop", [
            "Explanation is an iterative process.",
            "Check for understanding through reconstruction.",
            "Synthesis transforms information into knowledge."
        ])

        # Define Checklist
        checklist_items = ["1. Mathematically precise?", "2. Visual scaffold?", "3. Unbroken logic path?"]
        checklist = VGroup(*[Text(item, font_size=24, color=WHITE) for item in checklist_items])
        checklist.arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(checklist, 'A2', 'B5', scale_factor=0.6)

        # Proof Sketch
        pythagorean = MathTex(r"a^2 + b^2 = c^2", font_size=36, color=WHITE)
        self.place_at_grid(pythagorean, 'D3', scale_factor=1.0)
        pythagorean.set_opacity(0)
        
        pythagorean_label = Text("Theorem", font_size=24, color=WHITE)
        self.place_at_grid(pythagorean_label, 'D4', scale_factor=0.8)
        pythagorean_label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        # Display summary checklist
        self.play(Write(checklist))
        self.lecture[0].set_color("#FFFF00") # Highlight Line 1

        # === Animation for Lecture Line 2 ===
        # Highlight first two pillars
        checks = VGroup(*[Checkmark().scale(0.5).next_to(checklist[i], RIGHT) for i in range(2)])
        for c in checks: c.set_color("#00FF00")
        
        self.play(Create(checks[0]), Create(checks[1]))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Final check and display Pythagorean theorem
        last_check = Checkmark().scale(0.5).next_to(checklist[2], RIGHT).set_color("#00FF00")
        self.play(Create(last_check), FadeIn(pythagorean), FadeIn(pythagorean_label))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)

class Checkmark(VMobject):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_points_as_corners([
            [-0.2, 0, 0],
            [-0.05, -0.2, 0],
            [0.2, 0.2, 0]
        ])
        self.set_stroke(width=4)
