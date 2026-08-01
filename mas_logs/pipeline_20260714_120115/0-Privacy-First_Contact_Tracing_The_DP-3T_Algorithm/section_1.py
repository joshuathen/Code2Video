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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture lines
        self.setup_layout(
            "The Privacy-Security Paradox",
            [
                "Traditional contact tracing often relies on central surveillance.",
                "This risks exposing sensitive user location and identity.",
                "DP-3T provides a decentralized, privacy-first alternative approach."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line and show the tension between health (virus) and privacy (shield)
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create a red virus icon using simple shapes
        virus_core = Circle(radius=0.4, color="#FF5555", fill_opacity=0.5)
        virus_spikes = VGroup(*[
            Line(virus_core.get_center(), virus_core.get_center() + 0.6 * np.array([np.cos(a), np.sin(a), 0]), color="#FF5555")
            for a in np.linspace(0, 2*PI, 12, endpoint=False)
        ])
        virus = VGroup(virus_core, virus_spikes)
        self.place_at_grid(virus, "B2", scale_factor=0.8) # Avoid Column 1 (L010)
        
        # Create a green shield icon
        shield_shape = Polygon(
            [-0.4, 0.4, 0], [0.4, 0.4, 0], [0.4, -0.2, 0], [0, -0.6, 0], [-0.4, -0.2, 0],
            color="#55FF55", fill_opacity=0.5
        )
        shield = VGroup(shield_shape)
        self.place_at_grid(shield, "B5", scale_factor=0.8)
        
        self.play(FadeIn(virus), FadeIn(shield))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition to central surveillance visualization
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            FadeOut(virus),
            FadeOut(shield)
        )
        
        # Central white server icon
        server_box = Square(side_length=0.8, color=WHITE, fill_opacity=0.3)
        server_lines = VGroup(
            Line(LEFT*0.3, RIGHT*0.3, color=WHITE).shift(UP*0.2),
            Line(LEFT*0.3, RIGHT*0.3, color=WHITE),
            Line(LEFT*0.3, RIGHT*0.3, color=WHITE).shift(DOWN*0.2)
        )
        server = VGroup(server_box, server_lines)
        # Using narrower area for better centering (Issue 25 fix)
        self.place_in_area(server, "C3", "C4", scale_factor=0.8)
        
        # Multiple grey phone icons [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg]
        phone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg"
        # Balanced distribution, avoiding Row E (Issue 24 fix)
        phone_positions = ["B2", "B5", "D2", "D5"]
        phones = VGroup()
        for pos in phone_positions:
            p = SVGMobject(phone_path, color="#AAAAAA")
            # Increase scale for visibility (Issue 26 fix)
            self.place_at_grid(p, pos, scale_factor=0.8)
            phones.add(p)
            
        # Connect phones to central server with red lines
        red_lines = VGroup(*[
            Line(server.get_center(), p.get_center(), color="#FF5555", stroke_width=2)
            for p in phones
        ])
        
        self.play(FadeIn(server), FadeIn(phones), Create(red_lines))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition to decentralized DP-3T approach
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Connect phones directly to each other with thin blue lines (#5555FF)
        blue_connections = []
        for i in range(len(phones)):
            for j in range(i + 1, len(phones)):
                blue_connections.append(Line(phones[i].get_center(), phones[j].get_center(), color="#5555FF", stroke_width=2))
        blue_lines = VGroup(*blue_connections)
        
        # Remove central server and red lines, show mesh network
        self.play(
            FadeOut(server),
            FadeOut(red_lines),
            Create(blue_lines)
        )
        self.wait(2)
