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

class Section1Scene(TeachingScene):
    def construct(self):
        # Content from Shared State
        title_text = "The Classical Search Dilemma"
        lecture_lines = [
            "Finding a specific item in an unsorted database is slow.",
            "Classically, we must check every single item one by one.",
            "For N items, finding the target takes O(N) time."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors for visual matching
        BOX_COLOR = WHITE
        HIGHLIGHT_COLOR = "#FF0000"  # Red
        COMPLEXITY_COLOR = "#ADD8E6" # Light Blue
        INACTIVE_COLOR = GREY
        
        # Assets
        BOX_ASSET = "/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/box.svg"

        # === Animation for Lecture Line 1 ===
        # Display a grid of 10 identical boxes [Asset: box.svg] labeled 'Unsorted Database'.
        self.play(self.lecture[0].animate.set_color(BOX_COLOR))
        
        boxes = VGroup(*[SVGMobject(BOX_ASSET, color=BOX_COLOR).scale(0.3) for _ in range(10)])
        boxes.arrange_in_grid(2, 5, buff=0.2)
        # Using place_in_area as per layout requirements
        self.place_in_area(boxes, "C2", "D6")
        
        db_label = Text("Unsorted Database", font_size=20, color=BOX_COLOR)
        # Position label near the boxes (L003: within 1 grid unit)
        self.place_in_area(db_label, "B2", "B6")
        
        self.play(LaggedStartMap(FadeIn, boxes, shift=UP), Write(db_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight boxes one by one with a red (#FF0000) question mark appearing above them.
        self.play(
            self.lecture[0].animate.set_color(INACTIVE_COLOR),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Counter for boxes checked
        counter_label = Text("Boxes checked: ", font_size=18, color=WHITE)
        counter_val = Text("0", font_size=18, color=WHITE)
        counter_group = VGroup(counter_label, counter_val).arrange(RIGHT, buff=0.1)
        # Issue 28: Positioning counter_group across E2-E6 for better centering
        self.place_in_area(counter_group, "E2", "E6")
        self.add(counter_group)
        
        for i in range(len(boxes)):
            # Red question mark above the box
            q_mark = Text("?", color=HIGHLIGHT_COLOR, font_size=24)
            q_mark.next_to(boxes[i], UP, buff=0.1)
            
            # Update counter value
            new_counter_val = Text(str(i + 1), font_size=18, color=WHITE).move_to(counter_val)
            
            self.play(
                boxes[i].animate.set_color(HIGHLIGHT_COLOR),
                FadeIn(q_mark, scale=0.5),
                Transform(counter_val, new_counter_val),
                run_time=0.4
            )
            self.play(FadeOut(q_mark), run_time=0.1)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show 'Classical Complexity: O(N)' in light blue (#ADD8E6)
        self.play(
            self.lecture[1].animate.set_color(INACTIVE_COLOR),
            self.lecture[2].animate.set_color(COMPLEXITY_COLOR)
        )
        
        complexity_tex = Text("Classical Complexity: O(N)", font_size=20, color=COMPLEXITY_COLOR)
        # Issue 29: Scaling complexity_tex to 0.7 and placing in F2-F6
        self.place_in_area(complexity_tex, "F2", "F6", scale_factor=0.7)
        
        self.play(Write(complexity_tex))
        self.wait(2)
        
        # Wrap up
        self.play(self.lecture[2].animate.set_color(INACTIVE_COLOR))
        self.wait(2)
