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
        # Setup the layout with section title and lecture lines
        self.setup_layout("Summary and Real-World Connection", [
            "Matrix multiplication glues multiple geometric steps together.",
            "It \"bakes\" complex movements into a single matrix.",
            "This is essential for CGI and robotics."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Highlight current lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create sequence of matrices M3 * M2 * M1 = M
        # Using Text instead of MathTex to avoid 'latex' command dependency
        m1 = Text("M1", color="#88C0D0", slant=ITALIC) # Light Blue
        m2 = Text("M2", color="#A3BE8C", slant=ITALIC) # Light Green
        m3 = Text("M3", color="#EBCB8B", slant=ITALIC) # Light Yellow
        dot1 = Text("·", color=WHITE)
        dot2 = Text("·", color=WHITE)
        eq = Text("=", color=WHITE)
        m_final = Text("M", color="#B48EAD", slant=ITALIC) # Light Purple
        
        matrix_sequence = VGroup(m3, dot1, m2, dot2, m1, eq, m_final).arrange(RIGHT, buff=0.2)
        
        # Fix for Issue 36: Move to B2-B6 and scale to 1.2 to avoid crowding lecture lines
        self.place_in_area(matrix_sequence, "B2", "B6", scale_factor=1.2)
        
        # Animate writing the sequence
        self.play(
            LaggedStart(
                Write(m3), Write(dot1), Write(m2), Write(dot2), Write(m1),
                lag_ratio=0.3
            )
        )
        self.wait(1)
        
        # Show condensation into M
        product_vgroup = VGroup(m3, dot1, m2, dot2, m1)
        self.play(
            Write(eq),
            Transform(product_vgroup.copy(), m_final, remover=True),
            Write(m_final)
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to the second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Create a simple line-diagram robotic arm in #C0C0C0
        arm_color = "#C0C0C0"
        
        # Build relative to origin first for clean anchoring
        shoulder = Dot(ORIGIN, color=arm_color)
        upper_arm = Line(ORIGIN, RIGHT * 1.5, color=arm_color)
        elbow = Dot(RIGHT * 1.5, color=arm_color)
        forearm = Line(RIGHT * 1.5, RIGHT * 2.5, color=arm_color)
        wrist = Dot(RIGHT * 2.5, color=arm_color)
        hand = Line(RIGHT * 2.5, RIGHT * 3.0, color=arm_color)
        
        arm_elements = VGroup(shoulder, upper_arm, elbow, forearm, wrist, hand)
        
        # Fix for Issue 37: Anchor the robot arm to area D2-F6
        self.place_in_area(arm_elements, "D2", "F6", scale_factor=1.0)
        
        # Capture the actual base point after grid placement
        base_point = shoulder.get_center()
        
        self.play(Create(arm_elements))
        self.wait(1)
        
        # ValueTrackers for joint angles to simulate "baking" movements
        theta1 = ValueTracker(0)
        theta2 = ValueTracker(0)
        theta3 = ValueTracker(0)
        
        # Lengths of arm segments (match initial construction)
        L1, L2, L3 = 1.5, 1.0, 0.5
        
        # Efficient updater using pre-calculated points relative to anchored base_point
        def update_arm(mob):
            t1 = theta1.get_value() * DEGREES
            t2 = theta2.get_value() * DEGREES
            t3 = theta3.get_value() * DEGREES
            
            p0 = base_point
            p1 = p0 + np.array([np.cos(t1), np.sin(t1), 0]) * L1
            p2 = p1 + np.array([np.cos(t1 + t2), np.sin(t1 + t2), 0]) * L2
            p3 = p2 + np.array([np.cos(t1 + t2 + t3), np.sin(t1 + t2 + t3), 0]) * L3
            
            upper_arm.put_start_and_end_on(p0, p1)
            elbow.move_to(p1)
            forearm.put_start_and_end_on(p1, p2)
            wrist.move_to(p2)
            hand.put_start_and_end_on(p2, p3)

        arm_elements.add_updater(update_arm)
        
        # Animate the arm moving smoothly (simulating baked transformation)
        self.play(
            theta1.animate.set_value(-20),
            theta2.animate.set_value(45),
            theta3.animate.set_value(30),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to the third line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Complex movement showing multi-joint coordination (essential for robotics/CGI)
        self.play(
            theta1.animate.set_value(40),
            theta2.animate.set_value(-70),
            theta3.animate.set_value(-40),
            run_time=3,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.wait(2)
        
        # Cleanup
        arm_elements.remove_updater(update_arm)
        self.wait(1)
