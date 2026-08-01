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
        title_text = "Scalars vs. Vectors: The Treasure Map"
        lecture_lines = [
            "Scalars represent a magnitude or quantity, like distance.",
            "Vectors combine magnitude with a specific direction.",
            "A scalar tells us how much; vectors show where."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Scalars represent a magnitude or quantity, like distance.
        # Anim: A number '5' appears in white #FFFFFF, then a circle with radius 5 slowly pulses.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        num_5 = Text("5", font_size=48, color=WHITE)
        self.place_at_grid(num_5, "C2", scale_factor=1.0) # Issue 23 fix: move to C2
        
        circle = Circle(radius=1.5, color=WHITE, stroke_width=2)
        self.place_at_grid(circle, "C2", scale_factor=1.0) # Issue 23 fix: move to C2
        
        self.play(Write(num_5))
        self.play(Create(circle))
        self.play(circle.animate.scale(1.1), rate_func=there_and_back)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Vectors combine magnitude with a specific direction.
        # Anim: An arrow (vector) appears in yellow #FFFF00, pointing to the right, labeled '5 miles East'.
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        start_pos = self.grid["C2"]
        end_pos = self.grid["C6"] # Issue 21 fix: point to C6 for East direction
        vector_arrow = Arrow(start_pos, end_pos, color=YELLOW, buff=0)
        
        vector_label = Text("5 miles East", font_size=20, color=YELLOW)
        # Issue 22 fix: use place_in_area for better label positioning
        self.place_in_area(vector_label, 'D3', 'D5', scale_factor=0.6)
        
        self.play(GrowArrow(vector_arrow))
        self.play(Write(vector_label))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # A scalar tells us how much; vectors show where.
        # Anim: Dot (cat) moves in a chaotic circle around the origin C2, then straight to treasure at C6.
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Asset integration (Issue 19): Cat icon
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(cat, "C2", scale_factor=0.5)
        
        self.play(FadeIn(cat))
        
        # Chaotic movement around the origin
        path = Circle(radius=0.5, color=WHITE).move_to(self.grid["C2"])
        self.play(MoveAlongPath(cat, path), run_time=2, rate_func=linear)
        
        # Asset integration (Issue 19): Treasure icon
        treasure = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/treasure.svg")
        treasure.set_color(YELLOW)
        self.place_at_grid(treasure, "C6", scale_factor=0.8) # Issue 21 fix: place at C6
        
        self.play(FadeIn(treasure))
        
        # Movement along the vector
        self.play(cat.animate.move_to(self.grid["C6"]), run_time=1.5)
        
        # Final emphasis
        self.play(Flash(vector_arrow, color=YELLOW, flash_radius=0.5), run_time=0.5)
        self.play(Flash(vector_arrow, color=YELLOW, flash_radius=0.5), run_time=0.5)
        self.play(Indicate(self.lecture[2], color=YELLOW))
        
        self.wait(2)
