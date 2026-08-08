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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Multi-Head Attention: Parallel Perspectives", 
            [
                "Transformers use multiple \"heads\" to process data simultaneously.",
                "Each head focuses on a different type of relationship.",
                "One head tracks grammar while another tracks logic.",
                "Together, they build a rich, multi-layered understanding.",
                "Parallel processing makes the model both fast and smart."
            ]
        )
        
        # Colors for visual elements
        COLOR_HEAD1 = "#FF0000" # Red
        COLOR_HEAD2 = "#00FF00" # Green
        COLOR_HEAD3 = "#0000FF" # Blue
        COLOR_HIGHLIGHT = "#FFFF00"
        ASSET_HEAD = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/head.svg"

        # === Animation for Lecture Line 1 ===
        # Transformers use multiple "heads" to process data simultaneously.
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Central word "Attention"
        word_attention = Text("Attention", font_size=36)
        self.place_in_area(word_attention, "C4", "D4")
        
        # Three Heads [Asset: .../head.svg]
        head1 = SVGMobject(ASSET_HEAD).set_color(COLOR_HEAD1)
        head2 = SVGMobject(ASSET_HEAD).set_color(COLOR_HEAD2)
        head3 = SVGMobject(ASSET_HEAD).set_color(COLOR_HEAD3)
        
        self.place_at_grid(head1, "A4", scale_factor=0.5)
        self.place_at_grid(head2, "E2", scale_factor=0.5)
        self.place_at_grid(head3, "E6", scale_factor=0.5)
        
        # Labels for heads
        head_label1 = Text("Head 1", font_size=16, color=COLOR_HEAD1).next_to(head1, UP, buff=0.1)
        head_label2 = Text("Head 2", font_size=16, color=COLOR_HEAD2).next_to(head2, DOWN, buff=0.1)
        head_label3 = Text("Head 3", font_size=16, color=COLOR_HEAD3).next_to(head3, DOWN, buff=0.1)

        self.play(
            FadeIn(word_attention),
            FadeIn(head1), FadeIn(head2), FadeIn(head3),
            Write(head_label1), Write(head_label2), Write(head_label3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Each head focuses on a different type of relationship.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Rays scanning the word (Simplified using Polygons/Triangles)
        # Ray 1: head1 to word_attention
        ray1 = Triangle(color=COLOR_HEAD1, fill_opacity=0.2).scale(0.4)
        ray1.move_to((head1.get_center() + word_attention.get_center()) / 2)
        ray1.rotate(np.arctan2(word_attention.get_y() - head1.get_y(), word_attention.get_x() - head1.get_x()) - np.pi/2)

        # Ray 2: head2 to word_attention
        ray2 = Triangle(color=COLOR_HEAD2, fill_opacity=0.2).scale(0.4)
        ray2.move_to((head2.get_center() + word_attention.get_center()) / 2)
        ray2.rotate(np.arctan2(word_attention.get_y() - head2.get_y(), word_attention.get_x() - head2.get_x()) - np.pi/2)

        # Ray 3: head3 to word_attention
        ray3 = Triangle(color=COLOR_HEAD3, fill_opacity=0.2).scale(0.4)
        ray3.move_to((head3.get_center() + word_attention.get_center()) / 2)
        ray3.rotate(np.arctan2(word_attention.get_y() - head3.get_y(), word_attention.get_x() - head3.get_x()) - np.pi/2)

        self.play(Create(ray1), Create(ray2), Create(ray3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # One head tracks grammar while another tracks logic.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Labels Grammar, Logic, Tone with corrected positions
        label_grammar = Text("Grammar", font_size=20, color=COLOR_HEAD1)
        label_logic = Text("Logic", font_size=20, color=COLOR_HEAD2)
        label_tone = Text("Tone", font_size=20, color=COLOR_HEAD3)
        
        # Applying VideoCritic Fixes (Issues 40, 41, 42)
        self.place_at_grid(label_grammar, "A5", scale_factor=0.8)
        self.place_at_grid(label_logic, "F2", scale_factor=0.8)
        self.place_at_grid(label_tone, "F6", scale_factor=0.8)
        
        self.play(Write(label_grammar), Write(label_logic), Write(label_tone))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Together, they build a rich, multi-layered understanding.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Pulsing effect for rays to represent integration
        self.play(
            ray1.animate.set_opacity(0.6),
            ray2.animate.set_opacity(0.6),
            ray3.animate.set_opacity(0.6),
            word_attention.animate.set_color(COLOR_HIGHLIGHT),
            run_time=0.5
        )
        self.play(
            ray1.animate.set_opacity(0.2),
            ray2.animate.set_opacity(0.2),
            ray3.animate.set_opacity(0.2),
            word_attention.animate.set_color(WHITE),
            run_time=0.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Parallel processing makes the model both fast and smart.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        fast_smart = Text("Fast & Smart", font_size=24, color=YELLOW)
        self.place_at_grid(fast_smart, "B4") # Using B4 as it's vacant and centered-ish
        
        self.play(FadeIn(fast_smart))
        self.play(Indicate(fast_smart))
        
        self.wait(2)
