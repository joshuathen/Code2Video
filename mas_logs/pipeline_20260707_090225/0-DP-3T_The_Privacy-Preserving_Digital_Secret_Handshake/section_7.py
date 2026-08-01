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

class Section7Scene(TeachingScene):
    def construct(self):
        # Initialize the layout with updated mandatory strings
        self.setup_layout(
            "Summary: Security through Math", 
            [
                'The DP-3T protocol loops through broadcast, log, and match.', 
                'A privacy shield protects the entire decentralized process.', 
                'DP-3T ensures our safety and privacy coexist through math.'
            ]
        )

        # Colors
        FLOW_COLOR = "#5DADE2"  # Blue
        SHIELD_COLOR = "#58D68D" # Green
        TEXT_COLOR = "#FDFEFE"   # White

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(FLOW_COLOR)

        # Smartphone Asset at center
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/smartphone.svg]
        smartphone = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/smartphone.svg")
        smartphone.set_color(WHITE)
        self.place_in_area(smartphone, "C3", "D4", scale_factor=0.6)

        # Icons for circular flow
        # Broadcast Icon (B2)
        broadcast = VGroup(
            Dot(color=FLOW_COLOR),
            Arc(radius=0.15, angle=PI, color=FLOW_COLOR).rotate(PI/4),
            Arc(radius=0.3, angle=PI, color=FLOW_COLOR).rotate(PI/4)
        )
        broadcast_label = Text("Broadcast", font_size=16, color=FLOW_COLOR)
        
        # Log Icon (B5)
        log = VGroup(
            Rectangle(width=0.5, height=0.7, color=FLOW_COLOR),
            Line(start=[-0.15, 0.1, 0], end=[0.15, 0.1, 0], color=FLOW_COLOR),
            Line(start=[-0.15, -0.1, 0], end=[0.15, -0.1, 0], color=FLOW_COLOR)
        )
        log_label = Text("Log", font_size=16, color=FLOW_COLOR)

        # Upload Icon (E5)
        upload = VGroup(
            Circle(radius=0.25, color=FLOW_COLOR).shift(LEFT*0.1),
            Circle(radius=0.2, color=FLOW_COLOR).shift(RIGHT*0.2 + UP*0.1),
            Arrow(start=DOWN*0.3, end=UP*0.3, buff=0, color=FLOW_COLOR, stroke_width=2)
        )
        upload_label = Text("Upload", font_size=16, color=FLOW_COLOR)

        # Match Icon (E2)
        match = VGroup(
            Circle(radius=0.25, color=FLOW_COLOR).shift(LEFT*0.15),
            Circle(radius=0.25, color=FLOW_COLOR).shift(RIGHT*0.15),
        )
        match_label = Text("Match", font_size=16, color=FLOW_COLOR)

        # Positioning around the smartphone
        self.place_at_grid(broadcast, "B2", scale_factor=1.0)
        self.place_at_grid(broadcast_label, "A2", scale_factor=1.0)
        self.place_at_grid(log, "B5", scale_factor=1.0)
        self.place_at_grid(log_label, "A5", scale_factor=1.0)
        self.place_at_grid(upload, "E5", scale_factor=1.0)
        self.place_at_grid(upload_label, "F5", scale_factor=1.0)
        self.place_at_grid(match, "E2", scale_factor=1.0)
        self.place_at_grid(match_label, "F2", scale_factor=1.0)

        # Circular Flow Arrows
        arrow1 = Arrow(self.grid["B2"], self.grid["B5"], color=FLOW_COLOR, buff=0.5)
        arrow2 = Arrow(self.grid["B5"], self.grid["E5"], color=FLOW_COLOR, buff=0.5)
        arrow3 = Arrow(self.grid["E5"], self.grid["E2"], color=FLOW_COLOR, buff=0.5)
        arrow4 = Arrow(self.grid["E2"], self.grid["B2"], color=FLOW_COLOR, buff=0.5)

        self.play(
            Create(smartphone),
            Create(broadcast), Create(broadcast_label),
            Create(log), Create(log_label),
            Create(upload), Create(upload_label),
            Create(match), Create(match_label),
            run_time=1.5
        )
        self.play(Create(arrow1), Create(arrow2), Create(arrow3), Create(arrow4), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(SHIELD_COLOR)
        
        # Privacy Shield Asset
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg]
        shield = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/shield.svg")
        shield.set_color(SHIELD_COLOR)
        shield.set_fill(SHIELD_COLOR, opacity=0.3)
        # Issue 48: Use scale_factor=2.6
        self.place_in_area(shield, "A1", "F6", scale_factor=2.6) 

        shield_text = Text("Privacy Shield", font_size=24, color=SHIELD_COLOR)
        # Issue 46: Fix position at C3-C4
        self.place_in_area(shield_text, "C3", "C4", scale_factor=1.0)

        self.play(
            GrowFromCenter(shield),
            Write(shield_text),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(TEXT_COLOR)

        final_text = Text("DP-3T: Decentralized, Private, Safe", font_size=20, color=TEXT_COLOR, weight=BOLD)
        # Issue 47: Fix position at D3-D4
        self.place_in_area(final_text, "D3", "D4", scale_factor=0.8)
        
        # Fade background to highlight the final text
        self.play(
            VGroup(
                smartphone, broadcast, log, upload, match, 
                arrow1, arrow2, arrow3, arrow4, 
                broadcast_label, log_label, upload_label, match_label
            ).animate.set_opacity(0.1),
            shield_text.animate.set_opacity(0.4),
            Write(final_text),
            run_time=2
        )
        self.wait(3)
