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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing the Count", ["Each collision counts as a hit.", "Collisions increment a digital counter.", "Count approaches pi digits."])
        
        # Setup counter
        counter_val = ValueTracker(0)
        counter_label = Text("Collision Count: ", font_size=32)
        counter_number = Integer(0, font_size=32)
        counter_group = VGroup(counter_label, counter_number).arrange(RIGHT)
        # Apply fix from issue 20
        self.place_at_grid(counter_group, 'B4', scale_factor=0.8)
        
        # Setup blocks using assets
        block_a = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color=BLUE)
        block_b = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg", color=RED)
        
        # Apply fixes from issue 21 and 22
        self.place_at_grid(block_a, 'D2', scale_factor=1.2)
        self.place_at_grid(block_b, 'D5', scale_factor=1.2)
        
        # Update logic
        def update_number(mob):
            mob.set_value(int(counter_val.get_value()))
            
        counter_number.add_updater(update_number)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(counter_group), FadeIn(block_a), FadeIn(block_b))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Simulation
        for i in range(3):
            self.play(
                block_a.animate.shift(RIGHT * 1.5),
                run_time=0.5
            )
            counter_val.set_value(counter_val.get_value() + 1)
            self.play(
                counter_number.animate.set_color("#FFFF00"),
                run_time=0.2
            )
            self.play(
                counter_number.animate.set_color(WHITE),
                run_time=0.2
            )
            self.play(
                block_a.animate.shift(LEFT * 1.5),
                run_time=0.5
            )
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(1)
