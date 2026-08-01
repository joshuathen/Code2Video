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
        title_text = "The Privacy Dilemma"
        lecture_lines = [
            "Contact tracing helps stop virus spread effectively.",
            "Centralized tracking risks user privacy and surveillance.",
            "DP-3T offers a decentralized, privacy-first alternative."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        smartphone_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/smartphone.svg"
        server_path = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/server.svg"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        def create_stick_figure_with_phone(color, phone_side):
            head = Circle(radius=0.15, color=color)
            body = Line(DOWN * 0.15, DOWN * 0.5, color=color)
            arms = Line(LEFT * 0.25, RIGHT * 0.25, color=color).shift(DOWN * 0.25)
            legs = VGroup(
                Line(ORIGIN, DOWN * 0.3 + LEFT * 0.2, color=color),
                Line(ORIGIN, DOWN * 0.3 + RIGHT * 0.2, color=color)
            ).shift(DOWN * 0.5)
            
            phone = SVGMobject(smartphone_path, height=0.3)
            phone.set_color(WHITE)
            if phone_side == "right":
                phone.move_to(arms.get_right() + RIGHT * 0.1)
            else:
                phone.move_to(arms.get_left() + LEFT * 0.1)
                
            return VGroup(head, body, arms, legs, phone)

        alice = create_stick_figure_with_phone("#FFC0CB", "right")
        bob = create_stick_figure_with_phone("#ADD8E6", "left")
        
        alice_label = Text("Alice", font_size=16, color="#FFC0CB")
        bob_label = Text("Bob", font_size=16, color="#ADD8E6")

        # ISSUE 27: Alice at B1, Bob at B6, scale 0.8
        self.place_at_grid(alice, "B1", scale_factor=0.8)
        self.place_at_grid(bob, "B6", scale_factor=0.8)
        
        alice_label.next_to(alice, UP, buff=0.1)
        bob_label.next_to(bob, UP, buff=0.1)

        self.play(
            FadeIn(alice), FadeIn(alice_label),
            FadeIn(bob), FadeIn(bob_label)
        )
        
        # Move them towards each other to represent meeting
        alice_target_pos = self.grid["B3"]
        bob_target_pos = self.grid["B4"]
        
        self.play(
            alice.animate.move_to(alice_target_pos),
            alice_label.animate.next_to(alice_target_pos, UP, buff=0.1),
            bob.animate.move_to(bob_target_pos),
            bob_label.animate.next_to(bob_target_pos, UP, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line (RED for surveillance/risk)
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED)
        )

        # ISSUE 23: Central Server Asset
        # ISSUE 28: Server at D3-E4, scale 0.8
        server_svg = SVGMobject(server_path, height=1.0)
        server_svg.set_color("#FF0000")
        server_text = Text("Central Server", font_size=16, color="#FF0000").next_to(server_svg, DOWN, buff=0.1)
        server = VGroup(server_svg, server_text)
        self.place_in_area(server, "D3", "E4", scale_factor=0.8)
        
        # Dotted lines from server to Alice/Bob
        line_to_alice = DashedLine(server.get_top(), alice.get_bottom(), color="#FF0000")
        line_to_bob = DashedLine(server.get_top(), bob.get_bottom(), color="#FF0000")
        
        # Labels for surveillance data
        id_label = Text("Identity", font_size=14, color=WHITE)
        loc_label = Text("Location", font_size=14, color=WHITE)
        
        # Position labels near the dashed lines
        id_label.move_to(line_to_alice.get_center() + LEFT * 0.6)
        loc_label.move_to(line_to_bob.get_center() + RIGHT * 0.6)

        self.play(
            FadeIn(server),
            Create(line_to_alice),
            Create(line_to_bob)
        )
        self.play(FadeIn(id_label), FadeIn(loc_label))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third line (GREEN for privacy-first)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GREEN)
        )

        # Remove Centralized elements
        self.play(
            FadeOut(server), 
            FadeOut(line_to_alice), 
            FadeOut(line_to_bob), 
            FadeOut(id_label), 
            FadeOut(loc_label)
        )

        # DP-3T Shield Icon
        shield_shape = Polygon(
            [-0.3, 0.4, 0], [0.3, 0.4, 0], [0.3, -0.1, 0], [0, -0.5, 0], [-0.3, -0.1, 0],
            color="#00FF00", fill_opacity=0.3
        )
        shield_text = Text("DP-3T", font_size=14, color="#00FF00").move_to(shield_shape.get_center())
        shield = VGroup(shield_shape, shield_text)
        
        # ISSUE 29: Shield scale 0.6 at B3-B4
        self.place_in_area(shield, "B3", "B4", scale_factor=0.6) 

        # Interaction indicator (local bluetooth exchange)
        interaction = DoubleArrow(
            alice.get_right(), 
            bob.get_left(), 
            color="#00FF00", 
            buff=0.1,
            stroke_width=3
        )

        self.play(FadeIn(shield), Create(interaction))
        self.wait(3)
