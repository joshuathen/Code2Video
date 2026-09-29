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
        self.setup_layout("Defining Pi: The Mathematical Formalism", 
                          ["Pi is circumference divided by diameter.", 
                           "Big or small, the ratio stays constant.", 
                           "Pi is an endless, irrational number."])
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        tape = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tape.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display the formula C = πd
        formula = MathTex(r"C = \pi d", color="#FF00FF")
        self.place_in_area(formula, 'A2', 'B5', scale_factor=1.2)
        ruler_icon = self.place_at_grid(ruler.copy(), 'B6', scale_factor=0.5)
        self.play(Write(formula), FadeIn(ruler_icon))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        # Show d changing while π remains constant
        circle = Circle(radius=0.5, color="#00FFFF")
        self.place_at_grid(circle, 'E5', scale_factor=0.8)
        
        d_line = Line(start=circle.get_left(), end=circle.get_right(), color=WHITE)
        d_label = MathTex("d", color=WHITE).next_to(d_line, UP)
        tape_icon = self.place_at_grid(tape.copy(), 'E2', scale_factor=0.5)
        
        self.play(Create(circle), Create(d_line), Write(d_label), FadeIn(tape_icon))
        
        # Animate changing circle
        self.play(
            circle.animate.set_radius(1.0),
            d_line.animate.scale(2.0, about_point=circle.get_center()),
            run_time=2
        )
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Highlight the static ratio π with a glowing effect
        pi_val = MathTex(r"\pi \approx 3.14159...", font_size=48, color=YELLOW)
        self.place_at_grid(pi_val, 'D4', scale_factor=1.0)
        
        glow = SurroundingRectangle(pi_val, color=YELLOW, buff=0.1, fill_opacity=0.3)
        prot_icon = self.place_at_grid(protractor.copy(), 'D6', scale_factor=0.5)
        
        self.play(FadeIn(pi_val), Create(glow), FadeIn(prot_icon))
        self.play(Indicate(pi_val), run_time=2)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
