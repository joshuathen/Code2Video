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
        # Initialize the layout
        lines = [
            "Computers see pixels, but humans think in language.",
            "How do we bridge this multimodal gap mathematically?",
            "We map text and images into one shared space.",
            "Here, similar concepts sit close to each other.",
            "This unified map is the foundation of CLIP."
        ]
        self.setup_layout("The Core Concept: Bridging Text and Image", lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_A), run_time=1)
        # Visualizing pixels vs language concept
        pixel_grid = Square(side_length=1, fill_color="#333333", fill_opacity=1, stroke_color=WHITE)
        self.place_at_grid(pixel_grid, "B4", scale_factor=1.2)
        pixel_dots = VGroup(*[Square(side_length=0.1, fill_color=WHITE, fill_opacity=1) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.05)
        self.place_at_grid(pixel_dots, "B4")
        
        text_bubble = Text("A Red Apple", font_size=18, color=WHITE)
        self.place_at_grid(text_bubble, "D4")
        
        self.play(FadeIn(pixel_grid), FadeIn(pixel_dots), FadeIn(text_bubble))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_A),
            run_time=1
        )
        gap_arrow = DoubleArrow(pixel_grid.get_bottom(), text_bubble.get_top(), color=GREY, stroke_width=2)
        q_mark = Text("?", font_size=36, color=YELLOW)
        # Issue 37/54: Align q_mark at C4 to connect pixel grid (B4) and text bubble (D4)
        self.place_at_grid(q_mark, "C4")
        
        self.play(Create(gap_arrow), FadeIn(q_mark))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF5555"),
            FadeOut(pixel_grid, pixel_dots, text_bubble, gap_arrow, q_mark),
            run_time=1
        )
        
        # Create the shared space background
        shared_plane = Rectangle(width=5.5, height=5.5, fill_color="#333333", fill_opacity=1, stroke_color=GREY_E)
        self.place_in_area(shared_plane, "A1", "F6")
        plane_label = Text("2D Shared Space", font_size=14, color=GREY_A)
        self.place_at_grid(plane_label, "A6", scale_factor=0.8)
        
        # Apple text and image icon (Asset integration)
        apple_text = Text("Apple", font_size=24, color="#FFFFFF")
        # Issue 35/54: Move apple_text to B3 to avoid overlap with icon
        self.place_at_grid(apple_text, "B3")
        
        # Issue 29/54: Integrate apple icon asset
        apple_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/apple.svg").set_color("#FF5555")
        self.place_at_grid(apple_icon, "B4", scale_factor=0.6)
        
        self.play(FadeIn(shared_plane), FadeIn(plane_label))
        self.play(Write(apple_text), FadeIn(apple_icon))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00"),
            run_time=1
        )
        
        banana_text = Text("Banana", font_size=24, color="#FFFF00")
        self.place_at_grid(banana_text, "E2")
        
        dist_line = DashedLine(apple_text.get_center(), banana_text.get_center(), color=GREY_B)
        dist_label = Text("Semantic Distance", font_size=14, color=GREY_B)
        self.place_at_grid(dist_label, "D3", scale_factor=0.8)
        
        self.play(Write(banana_text))
        self.play(Create(dist_line), FadeIn(dist_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#00FF00"),
            FadeOut(dist_line, dist_label),
            run_time=1
        )
        
        # Issue 29/54: Glowing circle around Apple group
        glow_circle = Circle(radius=1.2, color="#00FF00", stroke_width=6)
        # Position it to encompass B3 and B4 area
        self.place_in_area(glow_circle, "B3", "B4")
        
        embedding_label = Text("Shared Embedding Space", font_size=20, color="#00FF00")
        # Issue 36/54: Expand embedding_label into C3 to C5 to avoid crowding
        self.place_in_area(embedding_label, "C3", "C5")
        
        self.play(Create(glow_circle))
        self.play(Write(embedding_label))
        self.wait(2)
