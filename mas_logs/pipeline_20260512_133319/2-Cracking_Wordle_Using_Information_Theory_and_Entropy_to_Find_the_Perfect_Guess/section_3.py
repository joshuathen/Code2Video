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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout(
            "The Feedback Mechanics: 243 Possible Universes",
            [
                'Consider the word "TRACE" as our initial guess.',
                'Five tiles and three colors create 243 possible patterns.',
                'We can visualize these patterns as buckets for words.',
                'The game sorts all potential solutions into these buckets.',
                'A specific pattern narrows the search to one bucket.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Display 'TRACE' with letters in gray (#808080) boxes
        word_trace = "TRACE"
        trace_boxes = VGroup()
        for i, char in enumerate(word_trace):
            box = Square(side_length=0.7, fill_color="#808080", fill_opacity=1, stroke_color=WHITE)
            letter = Text(char, font_size=36, color=WHITE)
            trace_boxes.add(VGroup(box, letter).arrange(ORIGIN))
        
        trace_boxes.arrange(RIGHT, buff=0.1)
        # Issue 43 Fix: Position trace_boxes at B1-B6
        self.place_in_area(trace_boxes, "B1", "B6", scale_factor=0.9)
        
        self.play(FadeIn(trace_boxes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Calculation '3^5 = 243'
        math_seq = Text("3^5 = 243", font_size=32, color=WHITE)
        self.place_in_area(math_seq, "C1", "C6", scale_factor=1.0)
        
        self.play(Write(math_seq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Create a 15x17 grid of tiny white empty squares
        buckets = VGroup()
        rows_count = 15
        cols_count = 17 # 15 * 17 = 255
        for _ in range(255):
            buckets.add(Square(side_length=0.08, stroke_width=1, stroke_color=WHITE))
        
        buckets.arrange_in_grid(rows=rows_count, cols=cols_count, buff=0.05)
        # Issue 42 Fix: Position buckets at E1-F6
        self.place_in_area(buckets, "E1", "F6", scale_factor=0.95)
        
        self.play(Create(buckets), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Animate a stream of words falling into buckets
        falling_words_list = ["CRANE", "SLATE", "ADIEU", "STARE", "AUDIO", "REACT", "PLATE", "POINT"]
        falling_word_mobs = VGroup(*[Text(w, font_size=12, color=WHITE) for w in falling_words_list])
        
        # Position words at the top (near row A)
        for i, word_mob in enumerate(falling_word_mobs):
            self.place_at_grid(word_mob, f"A{min(i+1, 6)}")
        
        # Animate falling into the grid area
        falling_animations = []
        for i, word_mob in enumerate(falling_word_mobs):
            target_idx = (i * 20) % 255
            target_pos = buckets[target_idx].get_center()
            falling_animations.append(word_mob.animate.move_to(target_pos).set_opacity(0))
            
        self.play(AnimationGroup(*[FadeIn(w) for w in falling_word_mobs], lag_ratio=0.1))
        self.play(AnimationGroup(*falling_animations, lag_ratio=0.1), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )
        
        # Zoom into one bucket (Green-Gray-Yellow-Gray-Gray)
        # Clear the top area to avoid overlap with zoom_group
        self.play(FadeOut(trace_boxes), FadeOut(math_seq), buckets.animate.set_opacity(0.3))
        
        big_bucket_pattern = VGroup()
        colors = ["#6AAA64", "#808080", "#C9B458", "#808080", "#808080"]
        for color in colors:
            big_bucket_pattern.add(Square(side_length=0.4, fill_color=color, fill_opacity=1, stroke_color=WHITE))
        big_bucket_pattern.arrange(RIGHT, buff=0.05)
        
        inside_words = VGroup(
            Text("TREAD", font_size=18),
            Text("TRAMS", font_size=18),
            Text("TRAPS", font_size=18),
            Text("TRACK", font_size=18),
            Text("TRASH", font_size=18)
        ).arrange(DOWN, buff=0.15)
        
        zoom_group = VGroup(big_bucket_pattern, inside_words).arrange(DOWN, buff=0.3)
        # Issue 41 Fix: Position zoom_group at B1-D6
        self.place_in_area(zoom_group, "B1", "D6", scale_factor=1.0)
        
        self.play(FadeIn(big_bucket_pattern), Write(inside_words))
        self.wait(3)

        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
