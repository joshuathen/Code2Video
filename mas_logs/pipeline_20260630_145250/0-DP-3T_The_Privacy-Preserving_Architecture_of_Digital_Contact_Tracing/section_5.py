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

class Section5Scene(TeachingScene):
    def construct(self):
        title_text = "Infection Reporting: The Decentralized Upload"
        lecture_lines = [
            "A diagnosed user receives a one-time authorization code.",
            "They upload only their secret daily master keys.",
            "The central server acts as a public bulletin board.",
            "The server never learns who the user met.",
            "Only keys from the infectious period are ever shared."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        PHONE_COLOR = "#5DADE2"
        CODE_COLOR = "#ECF0F1"
        KEY_COLOR = "#F39C12"
        BOARD_COLOR = "#ECF0F1"
        CAL_COLOR = "#BDC3C7"
        GRAY_DARK = "#95A5A6"
        HIGHLIGHT_COLOR = "#F1C40F"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(PHONE_COLOR)
        
        phone_body = RoundedRectangle(corner_radius=0.1, height=1.5, width=0.8, color=PHONE_COLOR, fill_opacity=0.2)
        phone_screen = Rectangle(height=1.2, width=0.7, color=PHONE_COLOR, fill_opacity=0.1).move_to(phone_body.get_center())
        phone = VGroup(phone_body, phone_screen)
        self.place_at_grid(phone, "D2")
        
        auth_code = Text("Code: 1234", font_size=18, color=CODE_COLOR)
        self.place_at_grid(auth_code, "D2")
        
        self.play(Create(phone), Write(auth_code))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(KEY_COLOR)
        
        secret_key = VGroup(
            Square(side_length=0.5, color=KEY_COLOR, fill_opacity=0.5),
            Text("SK", font_size=16, color=WHITE)
        )
        self.place_at_grid(secret_key, "D2")
        
        # Transition from code to key
        self.play(auth_code.animate.set_opacity(0), FadeIn(secret_key))
        self.wait(0.5)
        
        # Key starts moving towards the bulletin board area
        self.play(secret_key.animate.move_to(self.grid["C4"]), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BOARD_COLOR)
        
        board_box = Rectangle(height=2.8, width=2.4, color=BOARD_COLOR, fill_opacity=0.1)
        self.place_in_area(board_box, "B4", "D6")
        
        board_label = Text("Bulletin Board", font_size=20, color=BOARD_COLOR)
        # Fix Issue 43: scale_factor=0.8 for board_label at A5
        self.place_at_grid(board_label, "A5", scale_factor=0.8)
        
        self.play(Create(board_box), Write(board_label))
        # Move key into its list position on the board
        self.play(secret_key.animate.move_to(self.grid["B5"]).scale(0.8))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GRAY_DARK)
        
        # Add more keys to show a list of anonymous data on the board
        key2 = secret_key.copy()
        self.place_at_grid(key2, "C5")
        key3 = secret_key.copy()
        self.place_at_grid(key3, "D5")
        
        anon_text = Text("No identities or links", font_size=16, color=GRAY_DARK)
        # Fix Issue 45: scale_factor=0.7 for anon_text at E5
        self.place_at_grid(anon_text, "E5", scale_factor=0.7)
        
        self.play(FadeIn(key2), FadeIn(key3))
        self.play(Write(anon_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(CAL_COLOR)
        
        calendar_icon = VGroup(
            Square(side_length=0.6, color=CAL_COLOR, fill_opacity=0.2),
            Text("CAL", font_size=12, color=CAL_COLOR)
        )
        self.place_at_grid(calendar_icon, "B6")
        
        # Highlight infectious period window (approx 3 days)
        highlight_rect = SurroundingRectangle(VGroup(secret_key, key2, key3), color=HIGHLIGHT_COLOR, buff=0.15)
        window_label = Text("3-Day Window", font_size=16, color=HIGHLIGHT_COLOR)
        # Fix Issue 44: scale_factor=0.5 for window_label at C6
        self.place_at_grid(window_label, "C6", scale_factor=0.5)
        
        self.play(FadeIn(calendar_icon))
        self.play(Create(highlight_rect), Write(window_label))
        self.wait(2)
