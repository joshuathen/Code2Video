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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup the layout with the specific title and lecture lines
        self.setup_layout(
            "Summary and Conclusion", 
            [
                "Vectors are elements of any valid vector space.",
                "We moved from arrows to abstract rules.",
                "Linear algebra reveals these universal mathematical patterns."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Create a white vector arrow
        arrow = Arrow(start=LEFT, end=RIGHT, color=WHITE)
        # Apply Fix from Issue 40: Adjust area to prevent overlap
        self.place_in_area(arrow, 'B1', 'C2', scale_factor=0.8)
        
        # Create a white polynomial equation
        poly = MathTex(r"P(x) = a_n x^n + \dots + a_0", color=WHITE)
        # Apply Fix from Issue 41: Adjust area to prevent overlap
        self.place_in_area(poly, 'B3', 'C6', scale_factor=0.8)
        
        # Display them side-by-side
        self.play(Create(arrow), Write(poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Create a green frame labeled '8 Axioms'
        # Group arrow and poly to frame them
        vector_elements = VGroup(arrow, poly)
        frame = SurroundingRectangle(vector_elements, color="#00FF00", buff=0.3)
        
        axiom_label = Text("8 Axioms", color="#00FF00", font_size=24)
        # Apply Fix from Issue 42: Align and center label under frame
        self.place_in_area(axiom_label, 'D2', 'D5', scale_factor=1.0)
        
        self.play(Create(frame), Write(axiom_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Final summary text
        final_text = Text("Abstract Vector Spaces", color=WHITE, font_size=32)
        # Place in a central area of the grid
        self.place_in_area(final_text, "B2", "D5", scale_factor=1.2)
        
        # Fade out elements from previous steps and show final title
        self.play(
            FadeOut(arrow),
            FadeOut(poly),
            FadeOut(frame),
            FadeOut(axiom_label),
            Write(final_text)
        )
        self.wait(3)
