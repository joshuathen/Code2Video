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
        self.setup_layout("The Fundamental Theorem of Calculus (FTC)", [
            "FTC links integral to antiderivative.",
            "Distance equals position change difference.",
            "Differentiation reveals local slopes.",
            "Integration recovers original accumulation.",
            "Inverse relationship completes the theorem."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Write FTC formula
        ftc_formula = MathTex(r"\int_{a}^{b} f(x) \, dx = F(b) - F(a)", color="#ECF0F1")
        self.place_at_grid(ftc_formula, 'B3', scale_factor=1.0)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        self.place_at_grid(ruler, 'B1', scale_factor=0.3)
        self.play(Write(ftc_formula), FadeIn(ruler))
        self.play(self.lecture[0].animate.set_color("#ECF0F1"))

        # === Animation for Lecture Line 2 ===
        # Highlight fundamental theorem parts
        highlight_box = SurroundingRectangle(ftc_formula, color="#E74C3C", buff=0.1)
        self.play(Create(highlight_box))
        self.play(self.lecture[1].animate.set_color("#E74C3C"))

        # === Animation for Lecture Line 3 ===
        # Show inverse relationship with arrows
        arrow = Arrow(start=self.grid['C4'], end=self.grid['D4'], color="#F1C40F")
        label = Text("f'(x) = f(x)", font_size=24, color="#F1C40F")
        self.place_at_grid(label, 'D4', scale_factor=0.8)
        self.play(GrowArrow(arrow), Write(label))
        self.play(self.lecture[2].animate.set_color("#F1C40F"))

        # === Animation for Lecture Line 4 ===
        # Animate application of FTC
        app_text = Text("Area = F(b) - F(a)", font_size=24, color="#2ECC71")
        self.place_at_grid(app_text, 'F2', scale_factor=0.8)
        self.play(FadeIn(app_text))
        self.play(self.lecture[3].animate.set_color("#2ECC71"))

        # === Animation for Lecture Line 5 ===
        # Final display of theorem in frame
        frame = Rectangle(width=4, height=2, color="#3498DB")
        self.place_in_area(frame, 'B2', 'C5', scale_factor=0.9)
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'C6', scale_factor=0.3)
        self.play(Create(frame), FadeIn(compass))
        self.play(self.lecture[4].animate.set_color("#3498DB"))
        self.wait(1)
