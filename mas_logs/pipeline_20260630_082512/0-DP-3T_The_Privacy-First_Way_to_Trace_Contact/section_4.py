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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        self.setup_layout("Phase 2: The Diagnosis & Upload", [
            'If diagnosed, a user shares their daily keys.',
            'Only keys are uploaded, never location or contacts.',
            'This data is posted to a public board.',
            'The server remains blind to who met whom.'
        ])
        
        # Asset path
        phone_asset = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        
        # Define Colors for consistency with lecture highlights
        COLOR1 = YELLOW
        COLOR2 = RED
        COLOR3 = BLUE
        COLOR4 = GREEN
        
        # === Animation for Lecture Line 1 ===
        # Bolt's phone turns red; an arrow sends 'Daily Key' icon up.
        self.play(self.lecture[0].animate.set_color(COLOR1))
        
        bolt_phone = SVGMobject(phone_asset)
        self.place_at_grid(bolt_phone, "E2", scale_factor=0.8)
        bolt_phone.set_color(WHITE)
        bolt_label = Text("Bolt's Phone", font_size=16).next_to(bolt_phone, DOWN, buff=0.1)
        
        self.play(FadeIn(bolt_phone), FadeIn(bolt_label))
        self.play(bolt_phone.animate.set_color(RED))
        
        daily_key = VGroup(
            Square(side_length=0.4, fill_opacity=1, fill_color=COLOR1),
            Text("DK", font_size=14, color=BLACK)
        )
        self.place_at_grid(daily_key, "E2", scale_factor=1.0)
        
        up_arrow = Arrow(start=self.grid["E2"], end=self.grid["C2"], color=COLOR1, buff=0.4)
        
        self.play(Create(up_arrow), daily_key.animate.move_to(self.grid["C2"]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Visual strike-through over map and contact list icons on Bolt's phone.
        self.play(self.lecture[1].animate.set_color(COLOR2))
        
        # Location Icon
        loc_icon = VGroup(
            Circle(radius=0.15, color=WHITE),
            Dot(color=WHITE)
        ).next_to(bolt_phone, LEFT, buff=0.3)
        loc_label = Text("Location", font_size=12).next_to(loc_icon, UP, buff=0.1)
        
        # Contacts Icon
        contacts_icon = VGroup(
            Line(LEFT*0.15, RIGHT*0.15, color=WHITE),
            Line(LEFT*0.15, RIGHT*0.15, color=WHITE).shift(DOWN*0.1),
            Line(LEFT*0.15, RIGHT*0.15, color=WHITE).shift(DOWN*0.2)
        ).next_to(bolt_phone, RIGHT, buff=0.3)
        contacts_label = Text("Contacts", font_size=12).next_to(contacts_icon, UP, buff=0.1)
        
        self.play(FadeIn(loc_icon), FadeIn(loc_label), FadeIn(contacts_icon), FadeIn(contacts_label))
        
        strike_loc = Cross(loc_icon, stroke_color=RED, stroke_width=5)
        strike_contacts = Cross(contacts_icon, stroke_color=RED, stroke_width=5)
        
        self.play(Create(strike_loc), Create(strike_contacts))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The 'Daily Key' icon lands on a large 'Public Board' #A9A9A9.
        self.play(self.lecture[2].animate.set_color(COLOR3))
        
        public_board = RoundedRectangle(corner_radius=0.1, width=2.5, height=3, fill_opacity=0.3, fill_color="#A9A9A9", color=WHITE)
        self.place_in_area(public_board, "B4", "E6")
        board_label = Text("Public Board", font_size=20, color=WHITE).next_to(public_board, UP, buff=0.2)
        
        self.play(FadeIn(public_board), Write(board_label))
        self.play(daily_key.animate.move_to(self.grid["C5"]), FadeOut(up_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The board shows keys without any lines linking them to users.
        self.play(self.lecture[3].animate.set_color(COLOR4))
        
        # Add more keys to the board pool
        key_pool = VGroup()
        positions = ["B5", "D4", "D5", "D6"]
        for pos in positions:
            k = daily_key.copy()
            k.move_to(self.grid[pos])
            key_pool.add(k)
            
        self.play(FadeIn(key_pool))
        
        # User placeholders outside to emphasize the disconnect
        user_x = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(user_x, "A4")
        label_x = Text("User X?", font_size=12).next_to(user_x, UP, buff=0.1)
        
        user_y = Circle(radius=0.15, color=WHITE)
        self.place_at_grid(user_y, "F4")
        label_y = Text("User Y?", font_size=12).next_to(user_y, DOWN, buff=0.1)
        
        self.play(FadeIn(user_x), FadeIn(label_x), FadeIn(user_y), FadeIn(label_y))
        
        # A question mark to signify the server doesn't know the sources
        mystery_symbol = Text("?", font_size=48, color=COLOR4).move_to(public_board.get_center())
        self.play(Write(mystery_symbol))
        
        self.wait(2)
