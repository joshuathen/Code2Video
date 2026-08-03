from manim import *

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
        title = "Multi-Head Attention: Multiple Perspectives"
        lecture_lines = [
            "Single-head attention might miss complex linguistic nuances.",
            "Multi-head attention uses several parallel attention mechanisms.",
            "Each \"head\" focuses on different types of relationships.",
            "One head might track grammar, while another tracks logic.",
            "Combining these heads provides a deeper, multidimensional understanding."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        COLOR_TEXT = "#FFFFFF"
        COLOR_HEAD1 = "#FF0000"
        COLOR_HEAD2 = "#0000FF"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        sentence_str = ["The", "chef", "cooked", "the", "meal."]
        words = VGroup(*[Text(w, font_size=28, color=COLOR_TEXT) for w in sentence_str])
        words.arrange(RIGHT, buff=0.2)
        # Fix for Issue 32: Use C2-C6 and scale_factor 1.0 to avoid obstructing lecture notes
        self.place_in_area(words, 'C2', 'C6', scale_factor=1.0)
        
        self.play(FadeIn(words))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)
        
        head1_label = Text("Head 1 (Grammar)", font_size=18, color=COLOR_HEAD1)
        head2_label = Text("Head 2 (Logic)", font_size=18, color=COLOR_HEAD2)
        
        # Fix for Issue 33 and 34: Use B4 and D4 with scale_factor 0.8 to avoid overlap with lecture text
        self.place_at_grid(head1_label, 'B4', scale_factor=0.8)
        self.place_at_grid(head2_label, 'D4', scale_factor=0.8)
        
        self.play(Write(head1_label), Write(head2_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_HIGHLIGHT)
        
        # Spotlight 1 (Red) on 'chef' and 'cooked'
        spotlight1 = Ellipse(width=2.5, height=1.0, color=COLOR_HEAD1, fill_opacity=0.3, stroke_width=2)
        spotlight1.move_to(VGroup(words[1], words[2]).get_center())
        
        # Spotlight 2 (Blue) on 'cooked' and 'meal'
        spotlight2 = Ellipse(width=2.5, height=1.0, color=COLOR_HEAD2, fill_opacity=0.3, stroke_width=2)
        spotlight2.move_to(VGroup(words[2], words[4]).get_center())
        
        self.play(
            FadeIn(spotlight1),
            FadeIn(spotlight2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_HIGHLIGHT)
        
        # Subtle animation to show "multidimensional understanding"
        self.play(
            spotlight1.animate.scale(1.1).set_fill(opacity=0.5),
            spotlight2.animate.scale(1.1).set_fill(opacity=0.5),
            run_time=1
        )
        self.play(
            spotlight1.animate.scale(1/1.1).set_fill(opacity=0.3),
            spotlight2.animate.scale(1/1.1).set_fill(opacity=0.3),
            run_time=1
        )
        
        self.wait(2)
