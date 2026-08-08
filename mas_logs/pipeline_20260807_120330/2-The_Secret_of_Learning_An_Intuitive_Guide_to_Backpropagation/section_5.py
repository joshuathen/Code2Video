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
        # Setup Section 5
        title = "The Chain Rule: Connecting the Layers"
        lines = [
            "The Chain Rule connects local changes to final errors.",
            "We multiply sensitivities layer by layer moving backwards.",
            "This propagates the error signal through the entire network.",
            "It reveals how hidden layers impact the final outcome.",
            "Simple multiplication links all layers into one learning system."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SENSITIVITY = "#FFD700"
        COLOR_NODE = WHITE
        
        # === Animation for Lecture Line 1 ===
        # Show nodes A, B, and C connected by arrows in a row.
        self.play(self.lecture[0].animate.set_color(COLOR_SENSITIVITY))
        
        node_a = Circle(radius=0.4, color=COLOR_NODE)
        label_a = Text("A", font_size=24, color=COLOR_NODE)
        node_a_group = VGroup(node_a, label_a)
        self.place_at_grid(node_a_group, "C2")
        
        node_b = Circle(radius=0.4, color=COLOR_NODE)
        label_b = Text("B", font_size=24, color=COLOR_NODE)
        node_b_group = VGroup(node_b, label_b)
        self.place_at_grid(node_b_group, "C4")
        
        node_c = Circle(radius=0.4, color=COLOR_NODE)
        label_c = Text("C", font_size=24, color=COLOR_NODE)
        node_c_group = VGroup(node_c, label_c)
        self.place_at_grid(node_c_group, "C6")
        
        # Connections (Forward flow initially)
        arrow_ab = Arrow(self.grid["C2"] + RIGHT*0.4, self.grid["C4"] + LEFT*0.4, buff=0.1, color=WHITE)
        arrow_bc = Arrow(self.grid["C4"] + RIGHT*0.4, self.grid["C6"] + LEFT*0.4, buff=0.1, color=WHITE)
        
        self.play(
            Create(node_a_group),
            Create(node_b_group),
            Create(node_c_group)
        )
        self.play(Create(arrow_ab), Create(arrow_bc))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Label edges with 3x and 2x sensitivity values (#FFD700).
        self.play(
            self.lecture[0].animate.set_color(WHITE), 
            self.lecture[1].animate.set_color(COLOR_SENSITIVITY)
        )
        
        sens_ab = MathTex("3\\times", color=COLOR_SENSITIVITY)
        sens_bc = MathTex("2\\times", color=COLOR_SENSITIVITY)
        
        # Positioning labels above the arrows (Row B)
        # Resolved Issue 37: Increase scale to 1.0
        self.place_at_grid(sens_ab, "B3", scale_factor=1.0)
        self.place_at_grid(sens_bc, "B5", scale_factor=1.0)
        
        self.play(Write(sens_ab), Write(sens_bc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate a red signal pulsing backward from C to B then B to A.
        self.play(
            self.lecture[1].animate.set_color(WHITE), 
            self.lecture[2].animate.set_color(COLOR_SENSITIVITY)
        )
        
        # Backwards arrows to show direction of chain rule
        back_arrow_cb = Arrow(self.grid["C6"] + LEFT*0.4, self.grid["C4"] + RIGHT*0.4, buff=0.1, color=COLOR_SENSITIVITY)
        back_arrow_ba = Arrow(self.grid["C4"] + LEFT*0.4, self.grid["C2"] + RIGHT*0.4, buff=0.1, color=COLOR_SENSITIVITY)
        
        signal_pulse = Dot(color=COLOR_SENSITIVITY, radius=0.15).set_z_index(10)
        signal_pulse.move_to(self.grid["C6"])
        
        self.play(FadeIn(signal_pulse))
        self.play(
            signal_pulse.animate.move_to(self.grid["C4"]),
            Create(back_arrow_cb),
            sens_bc.animate.scale(1.2),
            run_time=1.5
        )
        self.play(
            signal_pulse.animate.move_to(self.grid["C2"]),
            Create(back_arrow_ba),
            sens_ab.animate.scale(1.2),
            run_time=1.5
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Display the calculation 3 * 2 = 6 appearing above the nodes.
        # Actually VideoCritic suggested Row D (below), which is better for layout.
        self.play(
            self.lecture[2].animate.set_color(WHITE), 
            self.lecture[3].animate.set_color(COLOR_SENSITIVITY)
        )
        
        # Calculation: 3 * 2 = 6
        # Resolved Issue 35: Move calculation to D3-D5, scale 1.2
        calculation = MathTex("3", "\\times", "2", "=", "6", color=COLOR_SENSITIVITY)
        self.place_in_area(calculation, "D3", "D5", scale_factor=1.2)
        
        self.play(Write(calculation))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Simple multiplication links all layers into one learning system.
        self.play(
            self.lecture[3].animate.set_color(WHITE), 
            self.lecture[4].animate.set_color(COLOR_SENSITIVITY)
        )
        
        # Resolved Issue 36: Move final sensitivity to E2-E6, scale 1.0
        final_sensitivity = MathTex("6\\times \\text{ Total Sensitivity (A to C)}", color=COLOR_SENSITIVITY)
        self.place_in_area(final_sensitivity, "E2", "E6", scale_factor=1.0)
        
        # Storyboard step 5: Highlight entire backward path A<-B<-C with a glow
        path_glow = VGroup(back_arrow_cb, back_arrow_ba, node_a_group, node_b_group, node_c_group)
        
        self.play(
            FadeIn(final_sensitivity, shift=UP),
            path_glow.animate.set_color(COLOR_SENSITIVITY)
        )
        self.play(Indicate(calculation[4]), Indicate(final_sensitivity))
        
        self.wait(3)
