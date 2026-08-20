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
        self.setup_layout("Summary & Conclusion", [
            "Independence simplifies probability calculations.",
            "Bayes' Theorem enables systematic belief updates.",
            "Probability dynamically updates as evidence arrives."
        ])
        
        # --- Define Elements ---
        # Magnifying glass and Stethoscope assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        stethoscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stethoscope.svg")
        
        # Belief Bar
        bar_bg = Rectangle(width=3, height=0.4, color=GRAY, fill_opacity=0.3)
        self.belief_bar_fill = Rectangle(width=0, height=0.4, color=BLUE, fill_opacity=0.8, stroke_width=0)
        self.belief_bar_fill.align_to(bar_bg, LEFT)
        belief_label = Text("Belief Level", font_size=20)
        
        # Bayes' Theorem formula (simplified)
        formula = MathTex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}", font_size=30)
        
        # Groupings
        bar_group = VGroup(bar_bg, self.belief_bar_fill, belief_label)
        summary_group = VGroup(formula, magnifying_glass, stethoscope)

        # Place elements (using fixes from critics)
        self.place_at_grid(belief_label, 'C2', scale_factor=0.7) # Line 65
        self.place_in_area(formula, 'D2', 'E5', scale_factor=0.6) # Line 66
        # Formula/label group is too large, placing as suggested
        self.place_in_area(summary_group, 'A4', 'F6', scale_factor=0.5) # Line 67

        # --- Animation for Lecture Line 1 ---
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(magnifying_glass))
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(FadeIn(formula), FadeIn(stethoscope))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Animate belief bar
        self.play(self.belief_bar_fill.animate.set_width(3), run_time=2)
        self.wait(2)
        
        # Fade out
        self.play(FadeOut(self.lecture), FadeOut(bar_bg), FadeOut(self.belief_bar_fill), 
                  FadeOut(belief_label), FadeOut(formula), FadeOut(self.title),
                  FadeOut(magnifying_glass), FadeOut(stethoscope))
