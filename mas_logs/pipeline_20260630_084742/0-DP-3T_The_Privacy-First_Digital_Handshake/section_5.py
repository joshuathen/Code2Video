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
        # Setup the layout with the required lecture lines
        self.setup_layout(
            "Sounding the Alarm: The Diagnosis Upload", 
            [
                "If Pip tests positive, he alerts the community anonymously.", 
                "He only uploads his daily seeds to a server.", 
                "The server broadcasts these keys without any identity data.", 
                "Pip's personal information remains safely on his own device.", 
                "No contact logs or locations are ever shared externally."
            ]
        )
        
        # Paths for assets
        PHONE_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        SERVER_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/server.svg"

        # === Animation for Lecture Line 1 ===
        # Pip's [Asset: phone.svg] turns red (#FF0000) and displays a 'Positive Test' icon.
        self.lecture[0].set_color("#FF0000")
        
        phone = SVGMobject(PHONE_ASSET, color=WHITE)
        self.place_at_grid(phone, "C2", scale_factor=0.6)
        
        positive_icon = Text("+", color="#FF0000", weight=BOLD)
        self.place_at_grid(positive_icon, "C2", scale_factor=0.8)
        
        self.play(DrawBorderThenFill(phone))
        self.wait(0.5)
        self.play(
            phone.animate.set_color("#FF0000"),
            Write(positive_icon)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pip's 'Secret Day Keys' (#FFA500) fly from his [Asset: phone.svg] to a 'Public Bulletin Board' (#FFFFFF).
        self.lecture[1].set_color("#FFA500")
        
        # Create server asset to act as the bulletin board
        server_board = SVGMobject(SERVER_ASSET, color=WHITE)
        self.place_at_grid(server_board, "C5", scale_factor=0.7)
        board_label = Text("Public Bulletin Board", font_size=16, color=WHITE)
        self.place_at_grid(board_label, "B5", scale_factor=1.0)
        
        # Secret keys represented as small orange circles
        keys = VGroup(*[
            Circle(radius=0.1, color="#FFA500", fill_opacity=1.0) for _ in range(3)
        ]).arrange(RIGHT, buff=0.1)
        self.place_at_grid(keys, "C2", scale_factor=0.5)
        
        self.play(Create(server_board), Write(board_label))
        self.play(
            keys.animate.move_to(self.grid["C5"]),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The Bulletin Board [Asset: server.svg] displays the keys with a label 'Anonymous Sick Keys'.
        self.lecture[2].set_color(WHITE)
        
        sick_label = Text("Anonymous Sick Keys", font_size=18, color="#FFA500")
        # Issue 46: Use area positioning for the long label
        self.place_in_area(sick_label, "D4", "D6", scale_factor=1.0)
        
        self.play(Write(sick_label))
        self.play(Indicate(keys, color="#FFA500"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A green 'Privacy Shield' (#00FF00) appears over Pip's identity on his [Asset: phone.svg].
        self.lecture[3].set_color("#00FF00")
        
        shield = Circle(radius=0.3, color="#00FF00", fill_opacity=0.3)
        shield_icon = Text("✓", color="#00FF00").scale(0.6)
        privacy_shield = VGroup(shield, shield_icon)
        # Issue 44: Privacy shield moved to B2
        self.place_at_grid(privacy_shield, "B2", scale_factor=1.0)
        
        identity_label = Text("Identity", font_size=14, color=WHITE)
        self.place_at_grid(identity_label, "B2", scale_factor=1.0)
        
        self.play(Write(identity_label))
        self.play(FadeIn(privacy_shield))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The text 'No Personal Data Uploaded' flashes on the screen (#FFFFFF).
        self.lecture[4].set_color(WHITE)
        
        no_data_text = Text("No Personal Data\nUploaded", color="#FFFFFF", font_size=20)
        # Issue 45: Move no_data_text to D2
        self.place_at_grid(no_data_text, "D2", scale_factor=0.8)
        
        for _ in range(3):
            self.play(FadeIn(no_data_text, run_time=0.3), FadeOut(no_data_text, run_time=0.3))
        
        self.play(FadeIn(no_data_text))
        self.wait(2)
