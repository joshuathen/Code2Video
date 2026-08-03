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

class Section4Scene(TeachingScene):
    def construct(self):
        title_str = "The Mathematical Dance: Dot Products & Softmax"
        lines = [
            "Math determines how much focus words give each other.",
            "A Dot Product measures the similarity between Q and K.",
            "Softmax converts these scores into weights that total 100%.",
            "This creates a \"weighted recipe\" of contextual meaning.",
            "Watch how 'crossed' focuses on 'robot' and 'street'."
        ]
        self.setup_layout(title_str, lines)

        # Colors
        COLOR_Q = "#0000FF"
        COLOR_K = "#FF0000"
        COLOR_SCORE = "#FFFFFF"
        COLOR_HEATMAP_HIGH = "#FFFF00"
        COLOR_HEATMAP_LOW = "#000033"
        COLOR_ARROWS = "#00FF00"
        COLOR_CONTEXT = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Math determines how much focus words give each other.
        self.lecture[0].set_color(BLUE)
        
        q_vec = Arrow(start=ORIGIN, end=RIGHT*1.2, color=COLOR_Q, buff=0, stroke_width=4)
        k_vec = Arrow(start=ORIGIN, end=UP*1.2, color=COLOR_K, buff=0, stroke_width=4)
        q_label = Text("Q", color=COLOR_Q, font_size=24)
        k_label = Text("K", color=COLOR_K, font_size=24)
        
        vectors = VGroup(q_vec, k_vec)
        self.place_at_grid(vectors, "C3")
        
        # Use updaters for labels to follow arrow ends
        q_label.add_updater(lambda m: m.next_to(q_vec.get_end(), RIGHT, buff=0.1))
        k_label.add_updater(lambda m: m.next_to(k_vec.get_end(), UP, buff=0.1))
        
        score_val = ValueTracker(0)
        score_label = Text("Similarity Score: ", color=COLOR_SCORE, font_size=20)
        score_num = DecimalNumber(0, color=COLOR_SCORE, font_size=20)
        score_num.add_updater(lambda d: d.set_value(score_val.get_value()))
        score_display = VGroup(score_label, score_num).arrange(RIGHT)
        self.place_at_grid(score_display, "C5")

        self.play(
            GrowArrow(q_vec), 
            GrowArrow(k_vec), 
            FadeIn(q_label), 
            FadeIn(k_label), 
            FadeIn(score_display)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A Dot Product measures the similarity between Q and K.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        
        self.play(
            Rotate(k_vec, angle=-PI/2, about_point=q_vec.get_start()),
            score_val.animate.set_value(0.98),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Softmax converts these scores into weights that total 100%.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(vectors, q_label, k_label, score_display))
        
        heatmap = VGroup()
        percentages = VGroup()
        for r in range(5):
            for c in range(5):
                color = COLOR_HEATMAP_LOW
                val_str = "0%"
                # Target cells (2,3) and (2,5) -> 0-indexed (1,2) and (1,4)
                if (r == 1 and c == 2):
                    color = COLOR_HEATMAP_HIGH
                    val_str = "45%"
                elif (r == 1 and c == 4):
                    color = COLOR_HEATMAP_HIGH
                    val_str = "40%"
                elif (r == 1 and c == 0): val_str = "8%"
                elif (r == 1 and c == 1): val_str = "5%"
                elif (r == 1 and c == 3): val_str = "2%"

                sq = Square(side_length=0.7, fill_color=color, fill_opacity=0.8, stroke_color=WHITE, stroke_width=1)
                heatmap.add(sq)
                
                p_text = Text(val_str, font_size=12, color=WHITE)
                percentages.add(p_text)
        
        heatmap.arrange_in_grid(rows=5, cols=5, buff=0.1)
        self.place_in_area(heatmap, "B3", "F6", scale_factor=0.6)
        
        for sq, pt in zip(heatmap, percentages):
            pt.move_to(sq.get_center())

        self.play(FadeIn(heatmap))
        self.play(Write(percentages))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This creates a \"weighted recipe\" of contextual meaning.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        
        self.play(FadeOut(heatmap, percentages))
        
        words = ["The", "robot", "crossed", "the", "street"]
        word_mobs = VGroup(*[Text(w, font_size=24) for w in words])
        
        self.place_at_grid(word_mobs[0], "B3") # The
        self.place_at_grid(word_mobs[1], "C3") # robot
        self.place_at_grid(word_mobs[2], "D4") # crossed
        self.place_at_grid(word_mobs[3], "E3") # the
        self.place_at_grid(word_mobs[4], "F3") # street
        
        arrows = VGroup(
            Arrow(word_mobs[1].get_right(), word_mobs[2].get_left(), color=COLOR_ARROWS, stroke_width=8),
            Arrow(word_mobs[4].get_right(), word_mobs[2].get_bottom(), color=COLOR_ARROWS, stroke_width=6),
            Arrow(word_mobs[0].get_right(), word_mobs[2].get_top(), color=COLOR_ARROWS, stroke_width=2),
            Arrow(word_mobs[3].get_right(), word_mobs[2].get_left(), color=COLOR_ARROWS, stroke_width=2)
        )
        
        self.play(FadeIn(word_mobs))
        self.play(Create(arrows))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Watch how 'crossed' focuses on 'robot' and 'street'.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_CONTEXT)
        
        context_icon = Star(n=5, color=COLOR_CONTEXT, fill_opacity=1).scale(0.3)
        self.place_at_grid(context_icon, "D5")
        
        particles = VGroup(*[Dot(color=COLOR_ARROWS, radius=0.04) for _ in range(12)])
        for i, p in enumerate(particles):
            arr = arrows[i % len(arrows)]
            p.move_to(arr.get_start())
            
        self.play(FadeIn(context_icon))
        
        self.play(
            *[p.animate.move_to(context_icon.get_center()).set_opacity(0) for p in particles],
            context_icon.animate.scale(2.5),
            run_time=2
        )
        self.wait(2)
