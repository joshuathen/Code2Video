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
        # 1. Setup
        self.setup_layout("Application: Why LLMs Feel Human", [
            "Attention allows models to maintain context over long texts.",
            "This enables coherent storytelling and complex code generation.",
            "It's the engine behind the intelligence of modern AI."
        ])
        
        # === Animation for Lecture Line 1 ===
        # A long text block scrolls with active connections. Color L1 to #FFFF00.
        words_list = [
            "The", "starship", "Enterprise", "drifted",
            "silently", "through", "the", "void",
            "awaiting", "the", "captain's", "next",
            "command", "to", "engage", "warp."
        ]
        words_vgroup = VGroup(*[Text(w, font_size=18, color=WHITE) for w in words_list])
        words_vgroup.arrange_in_grid(rows=4, cols=4, buff=0.4)
        
        # Start at column 2 to satisfy B021 (preserving column 1 as a gap)
        # Issue 43: scale_factor=0.7
        self.place_in_area(words_vgroup, 'B2', 'E5', scale_factor=0.7)
        
        # Active connections (lines between words)
        line1 = Line(words_vgroup[0].get_center(), words_vgroup[10].get_center(), color=BLUE_B, stroke_width=1.5, stroke_opacity=0.6)
        line2 = Line(words_vgroup[3].get_center(), words_vgroup[15].get_center(), color=BLUE_B, stroke_width=1.5, stroke_opacity=0.6)
        connections = VGroup(line1, line2)
        
        all_text_content = VGroup(words_vgroup, connections)
        
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.play(FadeIn(words_vgroup))
        self.play(Create(connections))
        
        # Subtle "scrolling" by shifting the group up
        self.play(all_text_content.animate.shift(UP * 0.4), run_time=2)

        # === Animation for Lecture Line 2 ===
        # Distant words are linked by glowing arcs (#FFD700). Color L2 to #FFFF00.
        arc1 = ArcBetweenPoints(
            words_vgroup[1].get_center(), 
            words_vgroup[14].get_center(), 
            angle=PI/3, 
            color="#FFD700", 
            stroke_width=3
        )
        
        arc2 = ArcBetweenPoints(
            words_vgroup[12].get_center(), 
            words_vgroup[2].get_center(), 
            angle=PI/3, 
            color="#FFD700", 
            stroke_width=3
        )
        
        glowing_arcs = VGroup(arc1, arc2)
        
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Create(glowing_arcs))
        self.play(glowing_arcs.animate.set_stroke(opacity=0.8), run_time=1)

        # === Animation for Lecture Line 3 ===
        # A pulsing 'Context Mastery' badge appears at the center. Color L3 to #FFFF00.
        badge_circle = Circle(radius=1.0, color="#FFFF00", fill_opacity=0.2, stroke_width=4)
        badge_text = Text("Context\nMastery", font_size=22, color="#FFFF00")
        badge = VGroup(badge_circle, badge_text)
        
        # Issue 44: place_in_area(badge, 'C3', 'D4', scale_factor=0.8)
        self.place_in_area(badge, 'C3', 'D4', scale_factor=0.8)
        
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Focus shift: fade out previous layers
        self.play(FadeOut(all_text_content, glowing_arcs), run_time=1)
        self.play(FadeIn(badge))
        
        # Pulsing effect using an updater as per instruction 10
        badge.initial_height = badge.height
        self.total_time = 0
        def pulse_updater(m, dt):
            self.total_time += dt
            m.scale_to_fit_height(badge.initial_height * (1 + 0.1 * np.sin(self.total_time * 4)))
        
        badge.add_updater(pulse_updater)
        
        self.wait(4)
        badge.clear_updaters()
