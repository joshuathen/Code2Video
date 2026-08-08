from manim import *
import numpy as np

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
        # Final lecture lines from storyboard
        title = "Conclusion: Why Pi?"
        lines = [
            "Total collisions equal the arc divided by the jump.",
            "The number of clacks counts the steps of pi.",
            "Physics and geometry unite in this perfect calculation."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # Lecture Line 1: Total collisions equal the arc divided by the jump.
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Animation 1: Display formula 'Collisions = floor(π / θ)' in white (#FFFFFF)
        # Issue 35: Position formula at 'C2' to 'D6', scale 1.0 to avoid high placement.
        formula = MathTex(r"\text{Collisions} = \lfloor \pi / \theta \rfloor", color="#FFFFFF")
        self.place_in_area(formula, 'C2', 'D6', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1.5)
        
        # === Animation for Lecture Line 2 ===
        # Lecture Line 2: The number of clacks counts the steps of pi.
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            FadeOut(formula)
        )
        
        # Block-collision visual at 'E3'
        m1 = Square(side_length=0.4, color=BLUE, fill_opacity=0.8)
        m2 = Square(side_length=0.4, color=RED, fill_opacity=0.8)
        block_visual = VGroup(m1, m2).arrange(RIGHT, buff=0.1)
        self.place_at_grid(block_visual, 'E3')
        
        # Circle-jump visual
        # Issue 36: Position circle container at 'B2' to 'D4', scale 0.9 to prevent crowding bottom labels.
        circle_visual = Circle(radius=1.0, color=WHITE)
        self.place_in_area(circle_visual, 'B2', 'D4', scale_factor=0.9)
        
        # Yellow counter (#FFFF00) at 'B6'
        counter_val = ValueTracker(0)
        counter_label = Text("Clacks:", font_size=20, color="#FFFF00")
        counter_num = DecimalNumber(0, num_decimal_places=0, color="#FFFF00", font_size=20)
        counter_group = VGroup(counter_label, counter_num).arrange(RIGHT, buff=0.2)
        self.place_at_grid(counter_group, 'B6')
        
        # Use updater for dynamic counter update
        counter_num.add_updater(lambda d: d.set_value(counter_val.get_value()))
        
        self.play(
            FadeIn(block_visual),
            Create(circle_visual),
            FadeIn(counter_group)
        )
        
        # Animate synchronous clacks and jumps
        num_clacks = 5
        theta = PI / num_clacks
        arcs = VGroup()
        
        for i in range(num_clacks):
            # Create the jump arc on the circle
            start_angle = PI - i * theta
            jump_arc = Arc(
                radius=circle_visual.width / 2,
                start_angle=start_angle,
                angle=-theta,
                color="#FFFF00",
                arc_center=circle_visual.get_center()
            )
            arcs.add(jump_arc)
            
            # Animate block bump and arc creation
            self.play(
                m1.animate.shift(RIGHT*0.2).set_rate_func(there_and_back),
                Create(jump_arc),
                counter_val.animate.set_value(i + 1),
                run_time=0.4
            )
            
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Lecture Line 3: Physics and geometry unite in this perfect calculation.
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Animation 3: Large gold pi symbol (#FFD700)
        # Issue 34: Position pi_sym at 'B5' to 'E6', scale 1.0 to avoid overlap/obscuring count.
        pi_sym = MathTex(r"\pi", color="#FFD700").scale(4)
        self.place_in_area(pi_sym, 'B5', 'E6', scale_factor=1.0)
        
        # Glow effect with several layers
        glow = VGroup(*[
            MathTex(r"\pi", color="#FFD700").scale(4 * (1.0 + 0.1*j)).set_opacity(0.3 - 0.05*j)
            for j in range(1, 5)
        ])
        for g in glow:
            g.move_to(pi_sym.get_center())

        self.play(
            FadeOut(block_visual),
            FadeOut(circle_visual),
            FadeOut(arcs),
            FadeOut(counter_group),
            FadeIn(pi_sym),
            FadeIn(glow)
        )
        
        # Final glow expansion animation
        self.play(
            pi_sym.animate.scale(1.15),
            glow.animate.scale(1.3).set_opacity(0),
            run_time=2,
            rate_func=slow_into
        )
        self.wait(3)
