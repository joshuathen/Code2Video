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
        # Fetching data from storyboard
        title = "Step 3: The Encounter Log"
        lines = [
            "Your phone keeps a private log of encountered IDs.",
            "This log stays local and is never uploaded.",
            "It only records proximity and duration of contact."
        ]
        self.setup_layout(title, lines)

        # Color constants
        LOG_TITLE_COLOR = "#FFFFFF"
        ID_COLOR = "#D3D3D3"
        CLOUD_COLOR = "#1E90FF"
        CROSS_COLOR = RED
        ACTIVE_HIGHLIGHT = YELLOW

        # Initially dim lecture lines
        for line in self.lecture:
            line.set_color(GRAY_D)

        # === Animation for Lecture Line 1 ===
        # Your phone keeps a private log of encountered IDs.
        self.lecture[0].set_color(LOG_TITLE_COLOR)
        
        # Phone screen setup
        phone_frame = RoundedRectangle(height=5.5, width=3.2, corner_radius=0.3, color=WHITE)
        phone_screen = Rectangle(height=4.8, width=2.9, fill_color=BLACK, fill_opacity=1, color=WHITE)
        phone_screen.move_to(phone_frame.get_center())
        
        log_header = Text("Bob's Encounter Log", font_size=20, color=LOG_TITLE_COLOR)
        log_header.next_to(phone_screen.get_top(), DOWN, buff=0.3)
        
        phone_group = VGroup(phone_frame, phone_screen, log_header)
        # Fix Issue 40: Moved phone_group lower to avoid overlap with title
        self.place_in_area(phone_group, "C3", "F4", scale_factor=0.8)
        
        self.play(Create(phone_frame), FadeIn(phone_screen), Write(log_header))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This log stays local and is never uploaded.
        self.lecture[0].set_color(GRAY_D)
        self.lecture[1].set_color(CLOUD_COLOR)
        
        # Log rows appear one by one
        log_data = [
            "ID: 4f2a...",
            "ID: 89b1...",
            "ID: a3c4...",
            "ID: 1e7f..."
        ]
        
        log_rows = VGroup(*[
            Text(line, font_size=16, color=ID_COLOR) for line in log_data
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        log_rows.next_to(log_header, DOWN, buff=0.5)
        
        for row in log_rows:
            self.play(Write(row), run_time=0.4)
        
        # Cloud Icon Construction
        c1 = Circle(radius=0.35, color=CLOUD_COLOR, fill_opacity=1).shift(LEFT*0.3)
        c2 = Circle(radius=0.45, color=CLOUD_COLOR, fill_opacity=1).shift(UP*0.2)
        c3 = Circle(radius=0.35, color=CLOUD_COLOR, fill_opacity=1).shift(RIGHT*0.3)
        cb = Rectangle(height=0.4, width=0.8, color=CLOUD_COLOR, fill_opacity=1).shift(DOWN*0.1)
        cloud_icon = VGroup(c1, c2, c3, cb)
        # Fix Issue 39: Move cloud icon to A5 to avoid overlap with phone log header
        self.place_at_grid(cloud_icon, "A5", scale_factor=0.6)
        
        cloud_label = Text("Central Server", font_size=18, color=CLOUD_COLOR)
        # Adjust label position to be above or below to ensure it fits better
        cloud_label.next_to(cloud_icon, DOWN, buff=0.2)
        
        # Red Slash (X mark)
        cross_line1 = Line(cloud_icon.get_corner(UL), cloud_icon.get_corner(DR), color=CROSS_COLOR, stroke_width=8)
        cross_line2 = Line(cloud_icon.get_corner(UR), cloud_icon.get_corner(DL), color=CROSS_COLOR, stroke_width=8)
        cross_mark = VGroup(cross_line1, cross_line2)
        
        self.play(FadeIn(cloud_icon), Write(cloud_label))
        self.play(Create(cross_mark))
        
        # Arrow pointing away from cloud or just staying local
        local_arrow = CurvedArrow(phone_frame.get_right() + UP*0.5, phone_frame.get_right() + DOWN*0.5, angle=-PI, color=WHITE)
        local_label = Text("STAYS LOCAL", font_size=14, color=WHITE).next_to(local_arrow, RIGHT, buff=0.1)
        
        self.play(Create(local_arrow), Write(local_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It only records proximity and duration of contact.
        self.lecture[1].set_color(GRAY_D)
        self.lecture[2].set_color(ACTIVE_HIGHLIGHT)
        
        # Add proximity/duration details to the log entries (simulated by updating text or adding next to them)
        details = VGroup(*[
            Text(" [2m, 15min]", font_size=14, color=ACTIVE_HIGHLIGHT) for _ in range(4)
        ])
        for i, detail in enumerate(details):
            detail.next_to(log_rows[i], RIGHT, buff=0.2)
            
        self.play(LaggedStart(*[FadeIn(d) for d in details], lag_ratio=0.3))
        
        # Highlighting the log
        surround = SurroundingRectangle(VGroup(log_rows, details), color=ACTIVE_HIGHLIGHT, buff=0.2)
        self.play(Create(surround))
        self.play(Indicate(VGroup(log_rows, details), color=ACTIVE_HIGHLIGHT))
        
        self.wait(2)
