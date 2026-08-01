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
        title = "The 8 Axioms: The Rules of the Game"
        lines = [
            "Eight formal axioms define what makes a vector space.",
            "Commutativity ensures the order of addition does not matter.",
            "Every space must contain a unique zero vector.",
            "Adding the zero vector leaves any object unchanged.",
            "These rules apply to any set passing the test."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_DEFAULT = "#CCCCCC"
        COLOR_COMM = "#FFD700"
        COLOR_ZERO = "#FF4500"
        COLOR_DIST = "#1E90FF"
        COLOR_VALID = "#32CD32"

        # === Animation for Lecture Line 1 ===
        # Eight formal axioms define what makes a vector space.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        axiom_names = [
            "Commutativity", "Associativity", "Additive Identity", 
            "Additive Inverse", "Distributivity (V)", "Distributivity (S)",
            "Scalar Assoc.", "Scalar Identity"
        ]
        
        axiom_mobjects = []
        # Positions: A1-A2, B1-B2, C1-C2, D1-D2, A2... wait.
        # Let's use A1, B1, C1, D1 for first 4 and A2, B2, C2, D2 for next 4.
        positions = ["A1", "B1", "C1", "D1", "A2", "B2", "C2", "D2"]
        
        for i, name in enumerate(axiom_names):
            txt = Text(f"{i+1}. {name}", font_size=18, color=COLOR_DEFAULT)
            self.place_at_grid(txt, positions[i])
            axiom_mobjects.append(txt)
            self.play(Write(txt), run_time=0.2)
        
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Commutativity ensures the order of addition does not matter.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_COMM),
            axiom_mobjects[0].animate.set_color(COLOR_COMM)
        )
        
        u_label = MathTex("u", color=RED)
        plus1 = MathTex("+")
        v_label = MathTex("v", color=BLUE)
        eq = MathTex("=")
        v_label2 = MathTex("v", color=BLUE)
        plus2 = MathTex("+")
        u_label2 = MathTex("u", color=RED)
        
        comm_formula = VGroup(u_label, plus1, v_label, eq, v_label2, plus2, u_label2).arrange(RIGHT, buff=0.2)
        self.place_in_area(comm_formula, "A4", "B6", scale_factor=0.9)
        
        self.play(Write(comm_formula))
        
        # Swap animation logic (creating targets to swap)
        u1_pos = u_label.get_center()
        u2_pos = u_label2.get_center()
        v1_pos = v_label.get_center()
        v2_pos = v_label2.get_center()
        
        self.play(
            u_label.animate.move_to(v1_pos),
            v_label.animate.move_to(u1_pos),
            u_label2.animate.move_to(v2_pos),
            v_label2.animate.move_to(u2_pos),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every space must contain a unique zero vector.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_ZERO),
            axiom_mobjects[2].animate.set_color(COLOR_ZERO) # Additive Identity
        )
        
        # Visualizing a zero point
        zero_dot = Dot(color=WHITE)
        self.place_at_grid(zero_dot, "C5", scale_factor=1.0)
        zero_label = MathTex("\\vec{0}", color=COLOR_ZERO)
        zero_label.next_to(zero_dot, DOWN, buff=0.1)
        
        vec_v = Arrow(start=ORIGIN, end=RIGHT+UP, color=BLUE, buff=0)
        self.place_at_grid(vec_v, "C4", scale_factor=0.7)
        v_text = MathTex("v", color=BLUE).next_to(vec_v, LEFT, buff=0.1)
        
        self.play(Create(zero_dot), Write(zero_label), GrowArrow(vec_v), Write(v_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Adding the zero vector leaves any object unchanged.
        # Resolve Issue 23: place_in_area(identity_formula, 'C4', 'D6', scale_factor=0.9)
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_ZERO)
        )
        
        identity_formula = MathTex("v + \\vec{0} = v", color=WHITE)
        # Using the area requested by Issue 23
        self.place_in_area(identity_formula, 'C4', 'D6', scale_factor=0.9)
        
        self.play(Write(identity_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # These rules apply to any set passing the test.
        # Storyboard: Highlight 'Distributive' (#1E90FF) and show c(u+v) = cu + cv
        # Resolve Issue 24: place_in_area(dist_formula, 'E4', 'F6', scale_factor=0.9)
        # Resolve Issue 25: place_at_grid(valid_label, 'F5', scale_factor=1.2)
        
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_VALID),
            axiom_mobjects[4].animate.set_color(COLOR_DIST) # Distributivity (V)
        )
        
        dist_formula = MathTex("c(u + v) = cu + cv", color=COLOR_DIST)
        self.place_in_area(dist_formula, 'E4', 'F6', scale_factor=0.9)
        
        self.play(Write(dist_formula))
        
        # Add checkmarks to all axioms
        checkmarks = []
        for i, mobj in enumerate(axiom_mobjects):
            check = Tex("$\\checkmark$", color=COLOR_VALID).scale(0.8)
            check.next_to(mobj, LEFT, buff=0.1)
            checkmarks.append(check)
            
        self.play(
            *[Create(c) for c in checkmarks],
            *[m.animate.set_color(COLOR_VALID) for m in axiom_mobjects],
            run_time=1.0
        )
        
        valid_label = Text("Valid Vector Space", color=COLOR_VALID)
        self.place_at_grid(valid_label, 'F5', scale_factor=1.2)
        
        self.play(Write(valid_label))
        self.wait(2)
