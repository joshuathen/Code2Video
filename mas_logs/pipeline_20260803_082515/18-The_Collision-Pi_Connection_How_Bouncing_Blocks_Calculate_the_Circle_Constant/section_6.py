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
        # Teaching Content
        lecture_lines = [
            "Every collision is a jump along the circle's arc.",
            "The jump distance depends on the specific mass ratio.",
            "We count how many jumps fit in the circle.",
            "This total count matches the digits of Pi perfectly.",
            "Collisions map directly to the circle's total circumference."
        ]
        self.setup_layout("Collisions as Arc Segments", lecture_lines)

        # Colors from Storyboard/Constraints
        COLOR_CIRCLE = "#00FFFF"  # Cyan
        COLOR_BOUNCE = "#FFFF00"  # Yellow
        COLOR_HIGHLIGHT = "#FF00FF" # Magenta (used for count)
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Every collision is a jump along the circle's arc.
        self.play(self.lecture[0].animate.set_color(COLOR_CIRCLE))
        
        # Create a cyan circle on the right side grid
        # Occupying space roughly B2 to E5
        circle = Circle(radius=1.8, color=COLOR_CIRCLE)
        self.place_in_area(circle, "B2", "E5")
        center = circle.get_center()
        radius = 1.8
        
        # Starting velocity point
        start_angle = 160 * DEGREES
        start_p = circle.point_at_angle(start_angle)
        start_dot = Dot(start_p, color=COLOR_WHITE)
        
        self.play(Create(circle))
        self.play(FadeIn(start_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The jump distance depends on the specific mass ratio.
        self.play(
            self.lecture[0].animate.set_color(COLOR_WHITE),
            self.lecture[1].animate.set_color(COLOR_BOUNCE)
        )
        
        # Visualizing the jump/theta relationship
        theta_formula = MathTex(r"\theta = 2\arctan\sqrt{\frac{m_1}{m_2}}", font_size=28, color=COLOR_BOUNCE)
        # Fix for Issue 35: Re-position theta_formula to area A4-A6
        self.place_in_area(theta_formula, 'A4', 'A6', scale_factor=0.8)
        
        # Integration of Asset for Issue 24
        mass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mass.svg", color=COLOR_BOUNCE)
        self.place_at_grid(mass_icon, "A3", scale_factor=0.5)
        
        # Initial jump (yellow line)
        jump_angle = 38 * DEGREES
        p1 = start_p
        p2 = circle.point_at_angle(start_angle - jump_angle)
        first_jump = Line(p1, p2, color=COLOR_BOUNCE)
        
        self.play(Write(theta_formula), FadeIn(mass_icon))
        self.play(Create(first_jump))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We count how many jumps fit in the circle.
        self.play(
            self.lecture[1].animate.set_color(COLOR_WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Collision Counter UI
        collision_val = Integer(1, color=COLOR_HIGHLIGHT)
        collision_label = Text("Collisions:", font_size=24, color=COLOR_WHITE)
        counter_grp = VGroup(collision_label, collision_val).arrange(RIGHT)
        self.place_at_grid(counter_grp, "A2")
        
        self.add(counter_grp)
        
        # Sequentially animate jumps
        current_p = p2
        current_angle = start_angle - jump_angle
        
        for i in range(2, 6):
            next_angle = current_angle - jump_angle
            next_p = circle.point_at_angle(next_angle)
            next_jump = Line(current_p, next_p, color=COLOR_BOUNCE)
            
            # Point label
            lbl = Text(str(i), font_size=18, color=COLOR_WHITE)
            # Position label slightly outside the circle point
            lbl.move_to(next_p + (next_p - center) * 0.2)
            
            self.play(
                Create(next_jump),
                collision_val.animate.set_value(i),
                FadeIn(lbl, scale=0.5),
                run_time=0.8
            )
            current_p = next_p
            current_angle = next_angle

        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This total count matches the digits of Pi perfectly.
        self.play(
            self.lecture[2].animate.set_color(COLOR_WHITE),
            self.lecture[3].animate.set_color(COLOR_WHITE) # Text doesn't specify color, keeping focus
        )
        
        pi_relation = MathTex(r"N \approx \frac{\pi}{\theta}", font_size=34, color=COLOR_WHITE)
        # Fix for Issue 34: Position pi_relation at F2
        self.place_at_grid(pi_relation, 'F2', scale_factor=0.8)
        
        self.play(Write(pi_relation))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Collisions map directly to the circle's total circumference.
        self.play(
            self.lecture[3].animate.set_color(COLOR_WHITE),
            self.lecture[4].animate.set_color(COLOR_CIRCLE)
        )
        
        # Highlight total circumference / total path
        # Illustrate that all these segments trace the circle
        trace_arc = Arc(
            radius=radius,
            start_angle=start_angle,
            angle=-jump_angle*5,
            arc_center=center,
            color=COLOR_CIRCLE,
            stroke_width=10
        )
        
        circum_text = Text("Circumference = 2π", font_size=24, color=COLOR_CIRCLE)
        # Fix for Issue 34: Position circum_text in area F3-F6
        self.place_in_area(circum_text, 'F3', 'F6', scale_factor=0.7)
        
        self.play(Create(trace_arc), FadeIn(circum_text), run_time=2.5)
        self.wait(3)
