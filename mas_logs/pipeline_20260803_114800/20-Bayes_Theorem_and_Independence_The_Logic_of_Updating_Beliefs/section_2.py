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
        # Data from storyboard and outline
        title_text = "Prerequisite: Conditional Probability & Sample Space Shrinking"
        lecture_lines = [
            "Dependency requires using conditional probability instead.",
            "Probability of A given B focuses on circle B.",
            "The sample space shrinks from everything to only B.",
            "We calculate the fraction of B that is A.",
            "This shift represents our updated knowledge."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        COLOR_A = "#0000FF" # Blue
        COLOR_B = "#00FF00" # Green
        COLOR_B_BRIGHT = "#90EE90" # Light Green
        COLOR_INTERSECT = "#FFFF00" # Yellow
        COLOR_UNIVERSE = WHITE
        
        # Set initial lecture line color to GRAY (dimmed) except the first one
        for i in range(1, len(self.lecture)):
            self.lecture[i].set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Show a large white rectangle (Universe). Place a blue circle (A) and a green circle (B) overlapping.
        self.lecture[0].set_color(WHITE)
        
        universe = Rectangle(height=4.0, width=5.0, color=COLOR_UNIVERSE)
        # FIX Issue 20: Position too high. Use B1-F6.
        self.place_in_area(universe, 'B1', 'F6', scale_factor=1.0)
        
        circle_a = Circle(radius=1.0, color=COLOR_A, fill_opacity=0.3)
        # FIX Issue 21: Circles too small. Scale 1.3.
        self.place_at_grid(circle_a, 'C3', scale_factor=1.3)
        
        circle_b = Circle(radius=1.0, color=COLOR_B, fill_opacity=0.3)
        # FIX Issue 21: Circles too small. Scale 1.3.
        self.place_at_grid(circle_b, 'D4', scale_factor=1.3)
        
        label_a = Text("A", font_size=24, color=COLOR_A)
        self.place_at_grid(label_a, 'B3')
        
        label_b = Text("B", font_size=24, color=COLOR_B)
        self.place_at_grid(label_b, 'E4')
        
        self.play(
            Create(universe),
            FadeIn(circle_a),
            FadeIn(circle_b),
            Write(label_a),
            Write(label_b),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Probability of A given B focuses on circle B.
        # Storyboard: Highlight circle B by making it brighter (#90EE90) while circle A and Universe dim.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_B),
            circle_b.animate.set_color(COLOR_B_BRIGHT).set_fill(opacity=0.5),
            circle_a.animate.set_stroke(opacity=0.2).set_fill(opacity=0.1),
            universe.animate.set_stroke(opacity=0.2),
            label_a.animate.set_fill(opacity=0.2),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The sample space shrinks from everything to only B.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_B_BRIGHT),
            run_time=0.5
        )
        
        # Calculate center of the area B1-F6
        grid_center = (self.grid['B1'] + self.grid['F6']) / 2
        
        self.play(
            FadeOut(universe),
            FadeOut(circle_a),
            FadeOut(label_a),
            circle_b.animate.scale(1.5).move_to(grid_center),
            label_b.animate.scale(1.2).move_to(self.grid['B5']), # Position label B near the top right of the circle
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We calculate the fraction of B that is A.
        # Highlight the intersection (A \cap B) in #FFFF00 and label it "P(A|B)".
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(COLOR_INTERSECT),
            run_time=0.5
        )
        
        # Calculate final state of A circle based on transformation applied to B
        # Rel pos from B to A was C3 - D4 = (-1, 1). 
        # After scaling B by 1.5, the relative position also scales.
        rel_pos = (self.grid['C3'] - self.grid['D4']) * 1.5
        new_radius = 1.3 * 1.5 # Initial radius was 1.0, scaled by 1.3 then by 1.5
        
        circle_a_final = Circle(radius=new_radius).move_to(grid_center + rel_pos)
        intersection_highlight = Intersection(
            circle_a_final, circle_b,
            color=COLOR_INTERSECT,
            fill_opacity=0.8
        ).set_stroke(COLOR_INTERSECT, width=3)
        
        label_formula = MathTex("P(A|B)", font_size=36, color=COLOR_INTERSECT)
        # FIX Issue 22: Label too large. Fix: scale_factor=0.7.
        self.place_at_grid(label_formula, 'C4', scale_factor=0.7)
        
        self.play(
            FadeIn(intersection_highlight),
            Write(label_formula),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This shift represents our updated knowledge.
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(WHITE),
            run_time=0.5
        )
        self.wait(2)
