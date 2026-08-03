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

class Section4TheMathematicalTrapScene(TeachingScene):
    def construct(self):
        # 1. Setup
        title = "The Prediction: What happens at n = 6?"
        lecture_lines = [
            "What happens when Max adds a sixth point?",
            "Intuition suggests the doubling pattern will surely continue.",
            "Do you expect to see thirty-two regions next?"
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from storyboard
        COLOR_N6 = "#FF0000"
        COLOR_MAX = "#FFD700"
        COLOR_FLASH = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Display n=6 in the table with a question mark (#FF0000).
        self.lecture[0].set_color(COLOR_N6)
        
        # Table structure for context (n=1 to 5)
        n_col = VGroup(Text("n", font_size=24, color=BLUE))
        r_col = VGroup(Text("Regions", font_size=24, color=BLUE))
        
        for n_val, r_val in [("1", "1"), ("2", "2"), ("3", "4"), ("4", "8"), ("5", "16")]:
            n_col.add(Text(n_val, font_size=22))
            r_col.add(Text(r_val, font_size=22))
        
        n_col.arrange(DOWN, buff=0.3)
        r_col.arrange(DOWN, buff=0.3)
        r_col.next_to(n_col, RIGHT, buff=1.0)
        table_vg = VGroup(n_col, r_col)
        
        # Position table in the upper-right area
        self.place_in_area(table_vg, "A4", "C6", scale_factor=0.9)
        
        self.play(FadeIn(table_vg))
        
        # Add the 'trap' row: n=6 and ? in Red
        n6_label = Text("6", font_size=22, color=COLOR_N6)
        r6_label = Text("?", font_size=22, color=COLOR_N6)
        n6_label.next_to(n_col, DOWN, buff=0.3)
        r6_label.next_to(r_col, DOWN, buff=0.3)
        n6_row = VGroup(n6_label, r6_label)
        
        self.play(Write(n6_row))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/max.svg] Max appears with a thought bubble showing '32?' (#FFD700).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_MAX)
        
        # Load Max asset (Issue 23)
        max_character = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/max.svg")
        max_label = Text("Max", font_size=18, color=COLOR_MAX).next_to(max_character, DOWN, buff=0.1)
        max_group = VGroup(max_character, max_label)
        
        # Position Max at the bottom right
        self.place_at_grid(max_group, "E5", scale_factor=0.8)
        
        # Thought bubble sequence
        bubble = Ellipse(width=1.2, height=0.9, color=WHITE, fill_opacity=0.1)
        bubble_text = Text("32?", font_size=26, color=COLOR_MAX)
        thought_bubble = VGroup(bubble, bubble_text)
        
        # Reposition thought_bubble to D4 (Issue 32)
        self.place_at_grid(thought_bubble, "D4", scale_factor=1.0)
        
        # Connecting dots for thought trail
        dot1 = Dot(radius=0.04, color=WHITE)
        self.place_at_grid(dot1, "E4", scale_factor=1.0)
        
        # Reposition dot2 to D5 (Issue 34)
        dot2 = Dot(radius=0.07, color=WHITE)
        self.place_at_grid(dot2, "D5", scale_factor=1.0)
        
        self.play(FadeIn(max_group, shift=UP))
        self.play(
            FadeIn(dot1),
            FadeIn(dot2),
            FadeIn(thought_bubble),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Text 'Predict the next number' flashes in white (#FFFFFF).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_FLASH)
        
        # Prediction flash
        predict_text = Text("Predict the next number", font_size=28, color=COLOR_FLASH)
        # Reposition predict_text to F3-F6 (Issue 33)
        self.place_in_area(predict_text, "F3", "F6", scale_factor=1.0)
        
        self.play(Flash(predict_text, color=COLOR_FLASH, line_length=0.2, num_lines=10))
        self.play(Write(predict_text))
        self.wait(3)
