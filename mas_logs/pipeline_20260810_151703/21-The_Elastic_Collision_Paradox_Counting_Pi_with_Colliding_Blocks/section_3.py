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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Mass ratios link collisions to digits of Pi.",
            "Ratio 1 to 1 gives 3 collisions.",
            "Ratio 1 to 100 gives 31 collisions.",
            "Ratio 1 to 10,000 gives 314 collisions.",
            "The number of collisions encodes Pi's digits."
        ]
        self.setup_layout("The Mathematical Mapping: Collision Count as Digits", lecture_lines)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        
        collision_text_raw = MathTex(r"3", r"1", r"4", font_size=64)
        
        # Groupings per requirements
        collision_count_group = VGroup(*[collision_text_raw[0], collision_text_raw[1], collision_text_raw[2]])
        grid_visual_container = VGroup(blocks, collision_count_group)
        
        # Applying grid layout requirements
        self.place_in_area(grid_visual_container, 'A1', 'F6', scale_factor=0.9)
        self.place_at_grid(collision_text_raw, 'D4', scale_factor=1.2)
        self.place_in_area(collision_count_group, 'C4', 'E5', scale_factor=1.0)
        
        # Initial state (hidden)
        for part in collision_text_raw:
            part.set_opacity(0)
            
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(FadeIn(collision_text_raw[0].set_color("#FFFF00")))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(FadeIn(collision_text_raw[1].set_color("#FFFF00")))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(FadeIn(collision_text_raw[2].set_color("#FFFF00")))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.play(Indicate(collision_count_group))
        self.wait(2)
