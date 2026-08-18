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
        self.setup_layout("Synthesis and Summary", [
            "Derivatives find rate of change.",
            "Integrals track total accumulation.",
            "They are two linked perspectives."
        ])
        
        # Applying requested layout changes
        self.place_at_grid(self.title, 'A3', scale_factor=1.0)
        self.place_in_area(self.lecture, 'A1', 'C1', scale_factor=0.8)

        # Assets
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        camera_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/camera.svg")

        # Animation Setup (Circle Group)
        clock_circle = Circle(radius=1.2, color=WHITE)
        hand = Line(ORIGIN, UP * 1.0, color=YELLOW)
        
        # Value tracker for hand rotation
        angle_tracker = ValueTracker(0)
        
        # Use persistent mobject for the sector (no always_redraw for complex stuff if possible)
        # But for this animation requirement, we use a simple Sector
        sector = Sector(
            arc_center=ORIGIN,
            radius=1.0,
            start_angle=PI/2,
            angle=0,
            color="#E74C3C",
            fill_opacity=0.5
        )
        
        clock_group = VGroup(clock_circle, hand, sector)
        self.place_in_area(clock_group, 'A2', 'E5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#3498DB"))
        self.place_at_grid(scale_icon, 'F2', scale_factor=0.5)
        self.play(Create(clock_circle), GrowFromCenter(hand), FadeIn(scale_icon))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#E74C3C"))
        
        # Manual update for animation
        def update_sector(s):
            angle = -angle_tracker.get_value()
            new_sector = Sector(
                arc_center=clock_group.get_center(),
                radius=1.0 * 0.9,
                start_angle=PI/2,
                angle=angle,
                color="#E74C3C",
                fill_opacity=0.5
            )
            s.become(new_sector)
            
        def update_hand(h):
            h.put_start_and_end_on(clock_group.get_center(), clock_group.get_center() + rotate_vector(UP * 1.0 * 0.9, -angle_tracker.get_value()))
            
        sector.add_updater(update_sector)
        hand.add_updater(update_hand)
        
        self.play(angle_tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        sector.remove_updater(update_sector)
        hand.remove_updater(update_hand)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        self.place_at_grid(camera_icon, 'F5', scale_factor=0.5)
        self.play(FadeIn(camera_icon))
        self.wait(1)
