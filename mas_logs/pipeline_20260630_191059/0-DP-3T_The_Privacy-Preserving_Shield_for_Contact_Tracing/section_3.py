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

class Section3Scene(TeachingScene):
    def construct(self):
        title = "The Daily Cycle: Generating Ephemeral IDs (EphID)"
        lines = [
            "Every day, your phone generates a Secret Day Key.",
            "From this key, it derives multiple Ephemeral IDs.",
            "These random IDs change every fifteen minutes.",
            "Frequent rotation prevents anyone from tracking your device.",
            "Your true identity remains hidden behind these IDs."
        ]
        self.setup_layout(title, lines)

        # Colors
        GOLD = "#FFD700"
        SKY_BLUE = "#87CEEB"
        WHITE_CLR = "#FFFFFF"
        RED_CLR = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Every day, your phone generates a Secret Day Key.
        self.play(self.lecture[0].animate.set_color(GOLD))
        
        sk_t_node = Circle(radius=0.5, color=GOLD, fill_opacity=0.3)
        sk_t_label = Text("SK_t", color=GOLD, font_size=24)
        sk_t_group = VGroup(sk_t_node, sk_t_label)
        # Fix Issue 32: Position sk_t_group in area B2-C4 to avoid title crowding.
        self.place_in_area(sk_t_group, 'B2', 'C4', scale_factor=0.8)
        
        self.play(FadeIn(sk_t_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # From this key, it derives multiple Ephemeral IDs.
        self.play(self.lecture[1].animate.set_color(SKY_BLUE))
        
        eph_ids = VGroup()
        # Positions adjusted for better layout below the key
        positions = ["D1", "D2", "D3", "D4"]
        arrows = VGroup()
        
        for i, pos in enumerate(positions):
            eph_node = RoundedRectangle(height=0.6, width=1.4, corner_radius=0.1, color=SKY_BLUE, fill_opacity=0.2)
            eph_label = Text(f"EphID_{i+1}", color=SKY_BLUE, font_size=20)
            eph_group = VGroup(eph_node, eph_label)
            self.place_at_grid(eph_group, pos)
            eph_ids.add(eph_group)
            
            arrow = Arrow(sk_t_group.get_bottom(), eph_group.get_top(), buff=0.1, color=WHITE, stroke_width=1.5)
            arrows.add(arrow)

        self.play(LaggedStart(*(GrowArrow(arr) for arr in arrows), lag_ratio=0.2))
        self.play(LaggedStart(*(FadeIn(eid) for eid in eph_ids), lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # These random IDs change every fifteen minutes.
        self.play(self.lecture[2].animate.set_color(WHITE_CLR))
        
        # Clock Icon to indicate time passing
        clock_circle = Circle(radius=0.3, color=WHITE_CLR)
        clock_hand = Line(clock_circle.get_center(), clock_circle.get_center() + UP * 0.25, color=WHITE_CLR)
        clock_icon = VGroup(clock_circle, clock_hand)
        self.place_at_grid(clock_icon, "B1", scale_factor=0.8)
        
        self.play(FadeIn(clock_icon))
        
        # Animate rotation of IDs as time passes
        for i in range(len(eph_ids)):
            self.play(
                clock_hand.animate.rotate(-PI/2, about_point=clock_circle.get_center()),
                eph_ids[i].animate.set_stroke(width=5, color=WHITE_CLR),
                run_time=0.4
            )
            if i > 0:
                self.play(eph_ids[i-1].animate.set_stroke(width=1, color=SKY_BLUE), run_time=0.2)
            self.wait(0.2)
        
        self.play(eph_ids[-1].animate.set_stroke(width=1, color=SKY_BLUE))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Frequent rotation prevents anyone from tracking your device.
        self.play(self.lecture[3].animate.set_color(RED_CLR))
        
        tracker_text = Text("TRACKER", color=RED_CLR, font_size=24)
        # Fix Issue 31: Place tracker_text at F3 to balance the frame.
        self.place_at_grid(tracker_text, 'F3', scale_factor=0.9)
        
        # Failed linking attempt
        link_arrows = VGroup()
        for i in range(len(eph_ids) - 1):
            l_arr = CurvedArrow(eph_ids[i].get_bottom(), eph_ids[i+1].get_bottom(), angle=-PI/4, color=RED_CLR, stroke_width=2)
            link_arrows.add(l_arr)
            
        self.play(FadeIn(tracker_text))
        self.play(LaggedStart(*(Create(l_arr) for l_arr in link_arrows), lag_ratio=0.4))
        
        cross = Cross(VGroup(tracker_text, link_arrows), color=RED_CLR)
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Your true identity remains hidden behind these IDs.
        self.play(self.lecture[4].animate.set_color(RED_CLR))
        
        phone_icon = Square(side_length=0.6, color=WHITE, fill_opacity=0.1)
        phone_label = Text("True ID", color=WHITE, font_size=18).next_to(phone_icon, UP, buff=0.1)
        phone = VGroup(phone_icon, phone_label)
        # Fix Issue 33: Place phone (True ID) at B5 to align with logic.
        self.place_at_grid(phone, 'B5', scale_factor=0.8)
        
        self.play(FadeIn(phone))
        shield = SurroundingRectangle(phone, color=GOLD, buff=0.2)
        self.play(Create(shield))
        
        self.wait(2)
