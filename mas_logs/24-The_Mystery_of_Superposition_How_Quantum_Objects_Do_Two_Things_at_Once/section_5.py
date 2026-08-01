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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lines
        title_str = "The Measurement Act: Wavefunction Collapse"
        lecture_lines = [
            'Superposition lasts only until we take a look.',
            'Measuring forces the system to choose one reality.',
            'The blurry wave suddenly collapses into a single point.',
            'Before opening the box, the cat is both states.',
            'Observation turns quantum mystery into classical fact.'
        ]
        self.setup_layout(title_str, lecture_lines)

        # Helper to create a cat silhouette using provided asset
        def create_cat_silhouette(color, label_text):
            # Issue 36: [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png]
            img = ImageMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/cat.png")
            img.height = 0.5
            label = Text(label_text, font_size=18, color=color)
            cat = Group(img, label)
            label.next_to(img, DOWN, buff=0.1)
            return cat

        # === Animation for Lecture Line 1 ===
        # Render a blurry, spread-out cyan wave (#00FFFF)
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # Create a "wavefunction" made of multiple overlapping sine waves
        wave_group = VGroup()
        for i in range(8):
            wave_layer = FunctionGraph(
                lambda x: 0.5 * np.sin(1.2 * x + i * 0.3),
                x_range=[-2, 2],
                color="#00FFFF",
                stroke_opacity=0.25
            )
            wave_group.add(wave_layer)
        
        self.place_in_area(wave_group, "B2", "E5")
        self.play(Create(wave_group), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Display two cat silhouettes as translucent overlays: one 'Sleeping' (#888888) and one 'Awake' (#FFFFFF)
        self.play(self.lecture[1].animate.set_color("#888888"))
        
        sleeping_cat = create_cat_silhouette("#888888", "Sleeping")
        sleeping_cat[0].set_opacity(0.4)
        sleeping_cat[1].set_opacity(0.4)
        
        awake_cat = create_cat_silhouette("#FFFFFF", "Awake")
        awake_cat[0].set_opacity(0.4)
        awake_cat[1].set_opacity(0.4)
        
        # Issue 47: Move sleeping_cat to B3
        self.place_at_grid(sleeping_cat, "B3", scale_factor=1.2)
        # Issue 48: Move awake_cat to E5
        self.place_at_grid(awake_cat, "E5", scale_factor=1.2)
        
        self.play(FadeIn(sleeping_cat), FadeIn(awake_cat))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # An eye icon representing an 'Observer' (#FFFF00) appears and looks toward the center.
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        
        eye_outer = Ellipse(width=0.7, height=0.35, color="#FFFF00", stroke_width=3)
        eye_iris = Dot(color="#FFFF00", radius=0.12)
        observer = VGroup(eye_outer, eye_iris)
        
        # Issue 49: Move observer to A4 and scale 0.8
        self.place_at_grid(observer, "A4", scale_factor=0.8)
        
        self.play(FadeIn(observer, shift=LEFT))
        # Look toward the center
        self.play(eye_iris.animate.shift(LEFT * 0.15), run_time=1)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # The spread-out wave instantly collapses into a single, sharp vertical white line (#FFFFFF).
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        center_pos = self.grid["D3"]
        sharp_line = Line(
            center_pos + UP * 0.7,
            center_pos + DOWN * 0.7,
            color="#FFFFFF",
            stroke_width=5
        )
        
        self.play(
            ReplacementTransform(wave_group, sharp_line),
            run_time=0.5,
            rate_func=exponential_decay
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The 'Sleeping' cat silhouette fades away completely, while 'Awake' cat becomes solid and fully opaque.
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        
        self.play(
            FadeOut(sleeping_cat),
            awake_cat[0].animate.set_opacity(1.0),
            awake_cat[1].animate.set_opacity(1.0),
            awake_cat.animate.scale(1.3).move_to(self.grid["C4"]),
            run_time=1.5
        )
        self.wait(2)
