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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The Geometric View: Vector Rotation"
        lines = [
            "We can visualize Grover's algorithm as a geometric rotation.",
            "The quantum state starts far from the target vector.",
            "Each iteration rotates the state toward the solution.",
            "The Oracle and Diffusion together create this rotation.",
            "The state vector gradually aligns with the target state."
        ]
        self.setup_layout(title, lines)

        # Colors
        AXIS_COLOR = WHITE
        STATE_COLOR = "#FF00FF"
        TARGET_COLOR = YELLOW
        ROTATION_COLOR = GREEN

        # === Animation for Lecture Line 1 ===
        # Display a 2D plane with white axes labeled |s> and |w>.
        axes = Axes(
            x_range=[0, 5],
            y_range=[0, 5],
            axis_config={"color": AXIS_COLOR, "include_tip": True},
            x_length=4,
            y_length=4,
        )
        
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.9)
        
        # Fixed: Using Text instead of MathTex to avoid FileNotFoundError for 'latex'
        state_label = Text("|s>", color=AXIS_COLOR)
        self.place_at_grid(state_label, 'E2', scale_factor=0.7)
        
        target_label = Text("|w>", color=AXIS_COLOR)
        self.place_at_grid(target_label, 'A5', scale_factor=0.7)

        self.play(self.lecture[0].animate.set_color(ROTATION_COLOR))
        self.play(Create(axes), Write(state_label), Write(target_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw the initial state vector |\psi\rangle in #FF00FF close to the |s\rangle axis.
        angle_theta = 15 * DEGREES
        origin = axes.get_origin()
        end_point = axes.c2p(4 * np.cos(angle_theta), 4 * np.sin(angle_theta))
        
        psi_vector = Arrow(
            start=origin,
            end=end_point,
            color=STATE_COLOR,
            buff=0
        )
        
        # Fixed: Using Text instead of MathTex
        psi_label = Text("|ψ>", color=STATE_COLOR)
        psi_label.next_to(psi_vector.get_end(), RIGHT, buff=0.1)

        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(STATE_COLOR))
        self.play(GrowArrow(psi_vector), Write(psi_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the vector rotating by an angle \theta toward the |w\rangle axis.
        new_angle = 45 * DEGREES
        
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(ROTATION_COLOR))
        
        rot_tracker = ValueTracker(angle_theta)
        
        # Performance optimization: Update in place
        def update_psi(mob):
            angle = rot_tracker.get_value()
            new_end = axes.c2p(4 * np.cos(angle), 4 * np.sin(angle))
            mob.put_start_and_end_on(origin, new_end)

        def update_label(mob):
            mob.next_to(psi_vector.get_end(), RIGHT, buff=0.1)

        psi_vector.add_updater(update_psi)
        psi_label.add_updater(update_label)
        
        # Arc to show theta
        arc = Arc(radius=1.0, start_angle=angle_theta, angle=new_angle-angle_theta, color=ROTATION_COLOR, arc_center=origin)
        # Fixed: Using Text instead of MathTex
        theta_label = Text("θ", color=ROTATION_COLOR).next_to(arc, UR, buff=0.1)

        self.play(
            rot_tracker.animate.set_value(new_angle),
            Create(arc),
            Write(theta_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Overlay a text box 'Oracle + Diffusion' to show the cause of rotation.
        box = SurroundingRectangle(self.lecture[3], color=ROTATION_COLOR, buff=0.1)
        info_rect = RoundedRectangle(corner_radius=0.1, height=1, width=3, fill_color=BLACK, fill_opacity=0.8, stroke_color=ROTATION_COLOR)
        info_text = Text("Oracle + Diffusion", font_size=20, color=ROTATION_COLOR)
        info_box = VGroup(info_rect, info_text)
        self.place_at_grid(info_box, 'C5', scale_factor=1.0)

        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(ROTATION_COLOR))
        self.play(Create(box), FadeIn(info_box))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # Show the vector landing almost perfectly on the |w\rangle axis.
        final_angle = 85 * DEGREES
        
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(TARGET_COLOR))
        self.play(
            rot_tracker.animate.set_value(final_angle),
            FadeOut(arc),
            FadeOut(theta_label),
            run_time=2
        )
        
        psi_vector.remove_updater(update_psi)
        psi_label.remove_updater(update_label)
        
        self.play(Indicate(psi_vector, color=TARGET_COLOR), Indicate(target_label, color=TARGET_COLOR))
        self.wait(3)
