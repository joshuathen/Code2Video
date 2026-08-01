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
        self.setup_layout("The Infection Report: Uploading the Secret Key", [
            "If infected, users voluntarily upload their Secret Day Keys.",
            "Only the keys are sent to the central server.",
            "The server never receives contact lists or locations."
        ])

        # === Animation for Lecture Line 1 ===
        # Bob icon turns red (#FF0000) and selects 14 'Secret Day Keys' (#FFD700).
        self.lecture[0].set_color("#FF0000")
        
        bob_head = Circle(radius=0.25, color=WHITE, fill_opacity=1)
        bob_body = RoundedRectangle(width=0.6, height=0.7, corner_radius=0.1, color=WHITE, fill_opacity=1).next_to(bob_head, DOWN, buff=0.05)
        bob = VGroup(bob_head, bob_body)
        self.place_at_grid(bob, "B2", scale_factor=0.8)
        self.add(bob)
        
        self.play(bob.animate.set_color("#FF0000"), run_time=1)
        
        keys = VGroup(*[Square(side_length=0.15, color="#FFD700", fill_opacity=1) for _ in range(14)])
        keys.arrange_in_grid(rows=2, cols=7, buff=0.1)
        self.place_at_grid(keys, "C2", scale_factor=0.6)
        
        self.play(Create(keys), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Keys (#FFD700) move to 'Cloud Server' [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cloud.svg] (#F0F8FF) and then to a 'Public Bulletin Board' (#F5F5DC).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFD700")
        
        # Cloud Server Icon (using Asset)
        cloud_svg = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cloud.svg").set_color("#F0F8FF")
        server_label = Text("Server", font_size=20, color="#F0F8FF").next_to(cloud_svg, UP, buff=0.1)
        server = VGroup(cloud_svg, server_label)
        self.place_at_grid(server, "B5", scale_factor=0.7)
        
        # Bulletin Board (Position updated to C5 per Issue 39)
        board_rect = Rectangle(width=1.4, height=1.0, color="#F5F5DC", fill_opacity=0.3, stroke_width=2)
        board_pin = Circle(radius=0.05, color=RED, fill_opacity=1).move_to(board_rect.get_top())
        board_label = Text("Bulletin Board", font_size=20, color="#F5F5DC").next_to(board_rect, UP, buff=0.1)
        bulletin_board = VGroup(board_rect, board_pin, board_label)
        self.place_at_grid(bulletin_board, "C5", scale_factor=0.8)
        
        self.play(FadeIn(server), FadeIn(bulletin_board))
        
        # Move keys to server
        self.play(keys.animate.move_to(self.grid["B5"]).scale(0.5), run_time=1.5)
        self.play(Indicate(server, color="#FFD700"), run_time=0.5)
        self.wait(0.5)
        
        # Move keys to bulletin board
        self.play(keys.animate.move_to(self.grid["C5"]).scale(1.5), run_time=1.5)
        self.play(Indicate(bulletin_board, color="#FFD700"), run_time=0.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Red 'X' marks over 'Contacts' and 'GPS' icons (#A9A9A9) to show exclusion.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF0000")
        
        # Simple Contacts Icon (Position updated to E2 per Issue 37)
        book = Rectangle(width=0.6, height=0.8, color="#A9A9A9", fill_opacity=0.2)
        lines = VGroup(*[Line(LEFT*0.2, RIGHT*0.2, color="#A9A9A9", stroke_width=2) for _ in range(3)]).arrange(DOWN, buff=0.15).move_to(book)
        contacts_label = Text("Contacts", font_size=20, color="#A9A9A9").next_to(book, DOWN, buff=0.1)
        contacts = VGroup(book, lines, contacts_label)
        self.place_at_grid(contacts, "E2", scale_factor=0.7)
        
        # Simple GPS Icon (Position updated to E4 per Issue 38)
        pin_head = Circle(radius=0.2, color="#A9A9A9", fill_opacity=1)
        pin_point = Triangle(color="#A9A9A9", fill_opacity=1).scale(0.2).rotate(PI).next_to(pin_head, DOWN, buff=0)
        gps_label = Text("GPS", font_size=20, color="#A9A9A9").next_to(pin_head, DOWN, buff=0.4)
        gps = VGroup(pin_head, pin_point, gps_label)
        self.place_at_grid(gps, "E4", scale_factor=0.7)
        
        self.play(FadeIn(contacts), FadeIn(gps))
        
        def make_x(pos):
            l1 = Line(pos + LEFT*0.4 + UP*0.4, pos + RIGHT*0.4 + DOWN*0.4, color="#FF0000", stroke_width=6)
            l2 = Line(pos + LEFT*0.4 + DOWN*0.4, pos + RIGHT*0.4 + UP*0.4, color="#FF0000", stroke_width=6)
            return VGroup(l1, l2)
            
        x_contacts = make_x(self.grid["E2"])
        x_gps = make_x(self.grid["E4"])
        
        self.play(Create(x_contacts), Create(x_gps), run_time=1)
        self.wait(2)
