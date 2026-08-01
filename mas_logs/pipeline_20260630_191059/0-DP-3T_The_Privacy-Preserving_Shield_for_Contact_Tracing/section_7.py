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

class Section7Scene(TeachingScene):
    def construct(self):
        # === LECTURE CONTENT ===
        title_str = "Summary: Why DP-3T Wins"
        lines_str = [
            "DP-3T protects identity through decentralized key rotation.",
            "Privacy is maintained by processing matches locally.",
            "Public health and personal secrets can coexist safely."
        ]
        self.setup_layout(title_str, lines_str)
        
        # COLORS
        MAILBOX_RED = "#B22222"
        SHIELD_GREEN = "#00FF7F"
        WHITE_TEXT = "#FFFFFF"
        USER_BLUE = "#1E90FF"
        
        # === Animation for Lecture Line 1 ===
        # Visual: Central Server icon becomes a simple 'Mailbox' (#B22222).
        self.play(self.lecture[0].animate.set_color(MAILBOX_RED))
        
        server_box = RoundedRectangle(corner_radius=0.1, height=0.7, width=0.9, color=MAILBOX_RED, fill_opacity=0.2)
        server_det = VGroup(*[Line(LEFT*0.2, RIGHT*0.2, color=MAILBOX_RED, stroke_width=2) for _ in range(2)]).arrange(DOWN, buff=0.1)
        server_icon = VGroup(server_box, server_det)
        # Issue 44 Fix: Position at A3, scale 0.8 to avoid vertical crowding
        self.place_at_grid(server_icon, "A3", scale_factor=0.8)
        
        server_lbl = Text("Central Authority", font_size=16, color=MAILBOX_RED)
        server_lbl.next_to(server_icon, UP, buff=0.1)
        
        mailbox_rect = Rectangle(height=0.5, width=0.7, color=MAILBOX_RED, fill_opacity=0.6)
        mailbox_flag = Line(ORIGIN, UP*0.2, color=RED, stroke_width=4).next_to(mailbox_rect, RIGHT, buff=0).shift(UP*0.1)
        mailbox_icon = VGroup(mailbox_rect, mailbox_flag)
        # Issue 45 Fix: Position at A3, scale 0.8 to avoid overlap with summary text
        self.place_at_grid(mailbox_icon, "A3", scale_factor=0.8)
        
        mailbox_lbl = Text("Dumb Mailbox", font_size=16, color=MAILBOX_RED)
        mailbox_lbl.next_to(mailbox_icon, UP, buff=0.1)
        
        self.play(FadeIn(server_icon), Write(server_lbl))
        self.wait(1)
        self.play(
            ReplacementTransform(server_icon, mailbox_icon),
            ReplacementTransform(server_lbl, mailbox_lbl)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual: User icons remain anonymous behind green shields (#00FF7F).
        self.play(self.lecture[1].animate.set_color(SHIELD_GREEN))
        
        user_locs = ["D1", "E2", "F3", "D5", "E6"]
        users = VGroup()
        shields = VGroup()
        
        for loc in user_locs:
            u = Circle(radius=0.18, color=USER_BLUE, fill_opacity=0.8)
            self.place_at_grid(u, loc)
            users.add(u)
            
            s_pts = [[0, 0.22, 0], [0.18, 0.1, 0], [0.18, -0.1, 0], [0, -0.22, 0], [-0.18, -0.1, 0], [-0.18, 0.1, 0]]
            s = Polygon(*s_pts, color=SHIELD_GREEN, fill_opacity=0.5, stroke_width=2)
            self.place_at_grid(s, loc)
            # Shields centered on users in rows D-F
            shields.add(s)
            
        self.play(LaggedStart(*[FadeIn(u) for u in users], lag_ratio=0.15))
        self.play(LaggedStart(*[GrowFromCenter(s) for s in shields], lag_ratio=0.15))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Visual: Text 'DP-3T: Public Health + Privacy' appears in white (#FFFFFF).
        self.play(self.lecture[2].animate.set_color(WHITE_TEXT))
        
        final_txt = Text("DP-3T: Public Health + Privacy", font_size=26, color=WHITE_TEXT)
        # Issue 43 Fix: Position at B1-B6, scale 1.1 to avoid overlap with user nodes in rows D-F
        self.place_in_area(final_txt, "B1", "B6", scale_factor=1.1)
        
        box = SurroundingRectangle(final_txt, color=WHITE, buff=0.2, stroke_width=1)
        
        self.play(Write(final_txt))
        self.play(Create(box))
        self.wait(3)
