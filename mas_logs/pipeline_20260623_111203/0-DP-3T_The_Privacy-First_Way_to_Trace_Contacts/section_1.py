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
        # Setup layout with title and lecture lines
        title_text = "The Privacy Dilemma"
        lecture_lines = [
            'Contact tracing stops pandemics by identifying exposure risks.',
            'But central tracking creates a "Big Brother" surveillance state.',
            'How can we notify users while protecting their identity?'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Asset paths
        eye_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/eye.svg"
        phone_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/phone.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#2ECC71"))
        
        park = Rectangle(width=4.0, height=2.5, fill_color="#2ECC71", fill_opacity=0.2, stroke_color="#2ECC71")
        self.place_in_area(park, "C2", "E5")
        park_label = Text("Park Area", font_size=16, color="#2ECC71").next_to(park, UP, buff=0.1)
        
        alice = Dot(color=WHITE).scale(1.5)
        self.place_at_grid(alice, "D2")
        alice_label = Text("Alice", font_size=14).next_to(alice, DOWN, buff=0.1)
        
        bob = Dot(color=WHITE).scale(1.5)
        self.place_at_grid(bob, "D5")
        bob_label = Text("Bob", font_size=14).next_to(bob, DOWN, buff=0.1)
        
        exposure_line = Line(alice.get_center(), bob.get_center(), color="#E74C3C", stroke_width=4)
        exposure_label = Text("Exposure Risk", font_size=16, color="#E74C3C")
        self.place_in_area(exposure_label, "C3", "C4")
        
        self.play(FadeIn(park), Write(park_label))
        self.play(FadeIn(alice), Write(alice_label), FadeIn(bob), Write(bob_label))
        self.play(Create(exposure_line), Write(exposure_label))
        self.play(Indicate(exposure_line, color="#E74C3C"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#95A5A6"))
        
        server = Square(side_length=0.8, color="#95A5A6", fill_opacity=0.5)
        self.place_in_area(server, "A3", "A4")
        server_label = Text("Central Server", font_size=14, color="#95A5A6").next_to(server, UP, buff=0.1)
        
        others = VGroup(*[Dot(color=GRAY_C, radius=0.05) for _ in range(4)])
        self.place_at_grid(others[0], "B1")
        self.place_at_grid(others[1], "B6")
        self.place_at_grid(others[2], "F1")
        self.place_at_grid(others[3], "F6")
        
        connections = VGroup(
            DashedLine(alice.get_center(), server.get_center(), color="#95A5A6"),
            DashedLine(bob.get_center(), server.get_center(), color="#95A5A6"),
            *[DashedLine(dot.get_center(), server.get_center(), color="#95A5A6", stroke_width=1) for dot in others]
        )
        
        eye_icon = SVGMobject(eye_path).set_color(WHITE)
        self.place_in_area(eye_icon, "A3", "A4", scale_factor=0.3)
        
        self.play(
            FadeOut(park), FadeOut(park_label), FadeOut(exposure_line), FadeOut(exposure_label),
            FadeIn(server), Write(server_label),
            FadeIn(others),
            Create(connections)
        )
        self.play(FadeIn(eye_icon))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#F1C40F"))
        
        cross_line1 = Line(eye_icon.get_left() + UP*0.2, eye_icon.get_right() + DOWN*0.2, color="#E74C3C", stroke_width=6)
        cross_line2 = Line(eye_icon.get_left() + DOWN*0.2, eye_icon.get_right() + UP*0.2, color="#E74C3C", stroke_width=6)
        red_x = VGroup(cross_line1, cross_line2)
        
        phone_icon = SVGMobject(phone_path).set_color(WHITE)
        self.place_at_grid(phone_icon, "E2", scale_factor=0.4)
        
        notif_rect = RoundedRectangle(corner_radius=0.1, width=0.6, height=0.4, color=WHITE)
        notif_text = Text("!", font_size=18, color=WHITE)
        notif_icon = VGroup(notif_rect, notif_text)
        self.place_at_grid(notif_icon, "E5", scale_factor=1.0)
        
        q_mark = Text("?", font_size=36, color="#F1C40F")
        self.place_in_area(q_mark, "E3", "E4")

        self.play(Create(red_x))
        self.play(
            FadeOut(server), FadeOut(server_label), FadeOut(others), FadeOut(connections),
            FadeIn(phone_icon), FadeIn(notif_icon), FadeIn(q_mark)
        )
        self.wait(2)
