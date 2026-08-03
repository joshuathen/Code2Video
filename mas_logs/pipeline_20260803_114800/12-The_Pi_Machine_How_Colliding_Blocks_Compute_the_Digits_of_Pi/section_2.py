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
        self.setup_layout("Prerequisite: The Laws of the 'Clack'", [
            "- Elastic collisions follow two fundamental laws of physics.",
            "- Momentum and energy are conserved in every impact.",
            "- No energy is lost as heat during these collisions."
        ])
        
        # Initial state: all lecture lines gray to emphasize highlights
        for line in self.lecture:
            line.set_color(GRAY)
        
        # === Animation for Lecture Line 1 ===
        # Display conservation equations for momentum and energy in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        mom_label = Text("Momentum:", font_size=18, color=WHITE)
        momentum_eq = MathTex(r"m v_1 + M v_2 = C_1", color=WHITE)
        
        en_label = Text("Energy:", font_size=18, color=WHITE)
        energy_eq = MathTex(r"\frac{1}{2} m v_1^2 + \frac{1}{2} M v_2^2 = C_2", color=WHITE)
        
        # Positioning using grid and area for clean layout, addressing Issues 22 and 23
        self.place_at_grid(mom_label, 'B2', scale_factor=0.6)
        self.place_in_area(momentum_eq, 'B4', 'B6', scale_factor=0.7)
        
        self.place_at_grid(en_label, 'C2', scale_factor=0.6)
        self.place_in_area(energy_eq, 'C4', 'C6', scale_factor=0.7)
        
        self.play(Write(mom_label), Write(momentum_eq))
        self.play(Write(en_label), Write(energy_eq))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the energy equation with a #FFFF00 pulse.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        self.play(
            Indicate(energy_eq, color="#FFFF00", scale_factor=1.1),
            Indicate(en_label, color="#FFFF00", scale_factor=1.1),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw v and V axes; plot a #00FFFF ellipse representing energy.
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"color": WHITE, "include_tip": True}
        )
        v1_label = MathTex("v_1", color=WHITE, font_size=18)
        v2_label = MathTex("v_2", color=WHITE, font_size=18)
        
        # Conservation of energy 1/2 mv1^2 + 1/2 Mv2^2 = E is an ellipse in v1, v2
        ellipse = Ellipse(width=3.0, height=1.2, color="#00FFFF")
        
        # Place graph group in area, addressing Issue 24
        graph_group = VGroup(axes, ellipse)
        self.place_in_area(graph_group, 'D2', 'F6', scale_factor=0.8)
        
        # Position labels relative to the placed axes
        v1_label.next_to(axes.x_axis.get_end(), DOWN, buff=0.1)
        v2_label.next_to(axes.y_axis.get_top(), LEFT, buff=0.1)
        
        self.play(Create(axes), Write(v1_label), Write(v2_label))
        self.play(Create(ellipse))
        self.wait(2)
