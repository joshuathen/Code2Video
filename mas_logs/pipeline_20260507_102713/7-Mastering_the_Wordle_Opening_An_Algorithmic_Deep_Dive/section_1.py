from manim import *
import random

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
        # Setup title and lecture lines
        title_text = "The Objective: Shrinking the Search Space"
        lecture_lines = [
            'Wordle is about narrowing down the word pool.',
            'Poor guesses like FUZZY eliminate very few words.',
            'Efficient guesses like TRACE prune the search space quickly.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Color change only for active line
        self.lecture[0].set_color(YELLOW)
        
        # A dense cloud of 2,315 small white (#FFFFFF) dots fills the screen to represent the word pool.
        num_dots = 2315
        dots = VGroup()
        random.seed(42) # Consistent random distribution
        
        # Grid boundaries: x [0.1, 5.9], y [-3.2, 2.6]
        for _ in range(num_dots):
            pos_x = random.uniform(0.1, 5.9)
            pos_y = random.uniform(-3.2, 2.6)
            dot = Dot(point=[pos_x, pos_y, 0], radius=0.015, color="#FFFFFF", fill_opacity=0.7)
            dots.add(dot)
        
        # Show the cloud population
        self.play(FadeIn(dots, lag_ratio=0.0001), run_time=2.0)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transition lecture highlight
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # The word 'FUZZY' [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/fuzz.svg] appears in grey (#787C7E) boxes
        fuzzy_word = self.create_wordle_word(
            "FUZZY", 
            "#787C7E", 
            asset_path="/mmfs1/data/home/jthen/Code2Video/assets/icon/fuzz.svg"
        )
        # Resolved Issue 29: scale_factor set to 0.7 to avoid overlap
        self.place_in_area(fuzzy_word, "A1", "A6", scale_factor=0.7)
        
        self.play(FadeIn(fuzzy_word, shift=UP*0.2))
        
        # "only a tiny fraction of dots vanish"
        num_fuzzy_removed = 75 
        all_indices = list(range(num_dots))
        random.shuffle(all_indices)
        fuzzy_indices = all_indices[:num_fuzzy_removed]
        remaining_indices = all_indices[num_fuzzy_removed:]
        
        dots_to_remove_fuzzy = VGroup(*[dots[i] for i in fuzzy_indices])
        self.play(FadeOut(dots_to_remove_fuzzy), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition lecture highlight
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # The word 'TRACE' [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/trace.svg] appears in grey (#787C7E) boxes
        trace_word = self.create_wordle_word(
            "TRACE", 
            "#787C7E", 
            asset_path="/mmfs1/data/home/jthen/Code2Video/assets/icon/trace.svg"
        )
        # Resolved Issue 30: scale_factor set to 0.7 to avoid overlap
        self.place_in_area(trace_word, "A1", "A6", scale_factor=0.7)
        
        # Contrast with previous guess
        self.play(FadeOut(fuzzy_word), FadeIn(trace_word))
        
        # "a massive portion of dots vanish instantly"
        num_remaining = len(remaining_indices)
        num_trace_removed = int(num_remaining * 0.93)
        trace_indices = remaining_indices[:num_trace_removed]
        
        dots_to_remove_trace = VGroup(*[dots[i] for i in trace_indices])
        self.play(FadeOut(dots_to_remove_trace, lag_ratio=0.005), run_time=1.5)
        self.wait(3)

    def create_wordle_word(self, word, box_color, asset_path=None):
        """Helper to create a row of Wordle-style boxes with letters and an optional icon."""
        boxes = VGroup()
        for letter in word:
            square = Square(side_length=0.6, fill_color=box_color, fill_opacity=1, stroke_width=2, stroke_color=WHITE)
            char = Text(letter, font_size=28, color=WHITE, weight=BOLD)
            boxes.add(VGroup(square, char))
        boxes.arrange(RIGHT, buff=0.1)
        
        if asset_path:
            try:
                icon = SVGMobject(asset_path).set_color(WHITE)
                # Ensure icon fits proportionally to the height of the wordle boxes
                icon.height = boxes.height * 0.9
                icon.next_to(boxes, LEFT, buff=0.3)
                return VGroup(icon, boxes)
            except:
                # Fallback if asset is missing in environment
                return boxes
            
        return boxes
