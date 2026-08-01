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

class Section7Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines from storyboard
        title_text = "Summary & Key Takeaways"
        lecture_lines = [
            "DP-3T provides security through decentralized data storage.",
            "Privacy is guaranteed by design, not just policy.",
            "We can track the virus without tracking the people."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors from storyboard
        SHIELD_COLOR = "#27AE60"
        HIGHLIGHT_COLOR = "#F1C40F"
        VIRUS_COLOR = "#E74C3C"
        SILHOUETTE_COLOR = "#BDC3C7"

        # === Animation for Lecture Line 1 ===
        # Color change for current line
        self.lecture[0].set_color(SHIELD_COLOR)
        
        # Large green shield (#27AE60) protects phone icons [Asset: .../phone.svg].
        shield = Polygon(
            [0, 1.2, 0], [1, 0.6, 0], [0.8, -0.8, 0], [0, -1.4, 0], [-0.8, -0.8, 0], [-1, 0.6, 0],
            color=SHIELD_COLOR, fill_opacity=0.3
        )
        # Fix from Issue 49: shield at B3-C6, factor 0.9
        self.place_in_area(shield, 'B3', 'C6', scale_factor=0.9)
        
        phones = VGroup()
        for _ in range(4):
            # Using phone asset
            phone_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/phone.svg", color=WHITE)
            phones.add(phone_asset)
        
        phones.arrange_in_grid(2, 2, buff=0.3)
        # Fix from Issue 49: phones at B4-C5, factor 0.7
        self.place_in_area(phones, 'B4', 'C5', scale_factor=0.7)

        self.play(DrawBorderThenFill(shield), FadeIn(phones))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color change for current line
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Highlights appear: 'Decentralized' (#F1C40F), 'Private' (#F1C40F).
        label_decentralized = Text("Decentralized", font_size=22, color=HIGHLIGHT_COLOR)
        label_private = Text("Private", font_size=22, color=HIGHLIGHT_COLOR)
        
        # Fix from Issue 49: labels at A3 and A6, factor 0.8
        self.place_at_grid(label_decentralized, 'A3', scale_factor=0.8)
        self.place_at_grid(label_private, 'A6', scale_factor=0.8)
        
        self.play(Write(label_decentralized), Write(label_private))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color change for current line
        self.lecture[2].set_color(VIRUS_COLOR)
        
        # A virus [Asset: .../virus.svg] chart (#E74C3C) peaks while silhouettes remain anonymous (#BDC3C7).
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 3, 1],
            x_length=3.5,
            y_length=1.8,
            axis_config={"include_tip": False, "color": GRAY}
        )
        # Fix from Issue 50: axes at D3-F6, factor 0.9
        self.place_in_area(axes, 'D3', 'F6', scale_factor=0.9)
        
        # Virus peak curve (Gaussian curve)
        curve = axes.plot(lambda x: 2.2 * np.exp(-0.8 * (x - 2.5)**2), color=VIRUS_COLOR)
        
        # Virus Asset
        virus_icon = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/virus.svg", color=VIRUS_COLOR)
        virus_icon.scale(0.3)
        # Place virus icon at the peak of the curve
        peak_pos = axes.c2p(2.5, 2.2)
        virus_icon.move_to(peak_pos + UP * 0.4)
        
        # Silhouettes (anonymous figures)
        silhouettes = VGroup()
        for _ in range(5):
            head = Circle(radius=0.08, color=SILHOUETTE_COLOR, fill_opacity=1)
            body = Polygon(
                [-0.15, -0.3, 0], [0.15, -0.3, 0], [0.12, -0.08, 0], [-0.12, -0.08, 0], 
                color=SILHOUETTE_COLOR, fill_opacity=1
            )
            silhouettes.add(VGroup(head, body.next_to(head, DOWN, buff=0.04)))
        
        silhouettes.arrange(RIGHT, buff=0.25)
        # Fix from Issue 50: silhouettes at F3-F6, factor 0.9
        self.place_in_area(silhouettes, 'F3', 'F6', scale_factor=0.9)

        self.play(
            Create(axes),
            Create(curve),
            FadeIn(virus_icon),
            FadeIn(silhouettes),
            # Fade out previous elements slightly to focus on the chart
            shield.animate.set_opacity(0.1),
            phones.animate.set_opacity(0.1),
            label_decentralized.animate.set_opacity(0.5),
            label_private.animate.set_opacity(0.5)
        )
        self.wait(2)
