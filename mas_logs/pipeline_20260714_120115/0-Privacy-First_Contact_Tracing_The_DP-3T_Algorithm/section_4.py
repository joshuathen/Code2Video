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
        # Data from storyboard and lecture lines
        title_text = "The Diagnosis Report (Infection Event)"
        lecture_lines = [
            "An infected user chooses to report their positive status.",
            "A public bulletin board is used to notify others anonymously.",
            "The user uploads only their daily Secret Keys.",
            "No names or location data are ever shared or stored.",
            "The board collects keys from all recently infected users."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors based on storyboard requirements
        COLOR_INFECTED = "#FF0000"  # Red
        COLOR_BOARD = "#FFFFFF"     # White
        COLOR_KEY = "#5555FF"       # Blue
        COLOR_PRIVATE = "#FF0000"   # Red (for X's)

        # Assets paths
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        BOARD_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/board.svg"

        # === Animation for Lecture Line 1 ===
        # Change 'Pixel's' phone icon [Asset: phone.svg] to red (#FF0000)
        pixel_phone = SVGMobject(PHONE_ASSET)
        pixel_phone.set_color(WHITE) # Start white
        pixel_label = Text("Pixel's Phone", font_size=16, color=WHITE).next_to(pixel_phone, UP, buff=0.1)
        pixel_group = VGroup(pixel_phone, pixel_label)
        # Use B2 as per instructions and VideoCritic feedback
        self.place_at_grid(pixel_group, "B2", scale_factor=0.8)
        
        self.play(FadeIn(pixel_group))
        self.play(
            self.lecture[0].animate.set_color(COLOR_INFECTED),
            pixel_phone.animate.set_color(COLOR_INFECTED),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A public bulletin board [Asset: board.svg] is used to notify others anonymously.
        board = SVGMobject(BOARD_ASSET)
        board.set_color(COLOR_BOARD)
        board_label = Text("Bulletin Board", font_size=18, color=COLOR_BOARD).next_to(board, UP, buff=0.1)
        board_group = VGroup(board, board_label)
        # B4-E6 is a large area for the board
        self.place_in_area(board_group, "B4", "E6", scale_factor=1.0)
        
        self.play(
            self.lecture[1].animate.set_color(COLOR_BOARD),
            FadeIn(board_group),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The user uploads only their daily Secret Keys.
        # Animate a blue 'Secret Key' (#5555FF) moving toward a 'Bulletin Board'.
        secret_key_label = Text("Secret Key", font_size=20, color=COLOR_KEY)
        # Issue 35: scale_factor=0.6 at B2 to fit inside the phone visual
        self.place_at_grid(secret_key_label, "B2", scale_factor=0.6)
        
        # Target position on the board
        target_pos = board.get_center()
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_KEY),
            FadeIn(secret_key_label),
            run_time=0.5
        )
        self.play(
            secret_key_label.animate.move_to(target_pos),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # No names or location data are ever shared or stored.
        # Place a red 'X' over text labels 'Name' and 'Location'.
        # Issue 33: Position at B3, scale 0.7
        privacy_content = VGroup(
            Text("Name", font_size=20, color=WHITE),
            Text("Location", font_size=20, color=WHITE)
        ).arrange(DOWN, buff=0.2)
        self.place_at_grid(privacy_content, "B3", scale_factor=0.7)
        
        # Using Cross for the red X
        cross = Cross(privacy_content, color=COLOR_PRIVATE, stroke_width=8)
        
        self.play(
            self.lecture[3].animate.set_color(COLOR_PRIVATE),
            FadeIn(privacy_content),
            Create(cross),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The board collects keys from all recently infected users.
        # Display the white Bulletin Board filling up with multiple blue 'Infected Key' labels.
        # Issue 34: other_keys at C4-E6, scale 0.7
        other_keys = VGroup(*[
            Text(f"Key {i}", font_size=16, color=COLOR_KEY) for i in range(1, 7)
        ]).arrange_in_grid(rows=3, cols=2, buff=0.3)
        
        self.place_in_area(other_keys, "C4", "E6", scale_factor=0.7)
        
        self.play(
            self.lecture[4].animate.set_color(COLOR_KEY),
            FadeOut(secret_key_label),
            FadeOut(privacy_content),
            FadeOut(cross),
            FadeIn(other_keys),
            run_time=1
        )
        self.wait(2)
