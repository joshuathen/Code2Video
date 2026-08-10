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
        lecture_lines = ["Consider two blocks sliding toward a wall.", "Small block m hits large block M.", "Collisions produce digits of Pi. Amazing!"]
        self.setup_layout("The Hook: An Unexpected Discovery", lecture_lines)
        
        # Load assets
        block_m_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        block_M_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        
        block_m = SVGMobject(block_m_path, color="#4169E1", fill_opacity=0.6)
        block_M = SVGMobject(block_M_path, color="#FF4500", fill_opacity=0.6)
        
        # Add labels
        label_m = Text("m", color=WHITE, font_size=24).next_to(block_m, UP)
        label_M = Text("M", color=WHITE, font_size=24).next_to(block_M, UP)
        
        m_group = VGroup(block_m, label_m)
        M_group = VGroup(block_M, label_M)
        
        # Collision count
        collision_count = Text("Collisions: 0", color=WHITE, font_size=24)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(m_group, 'C3', 'F6', scale_factor=0.9)
        self.place_in_area(M_group, 'C3', 'F6', scale_factor=1.1)
        # Offset them to be side-by-side
        m_group.shift(LEFT * 1.5)
        M_group.shift(RIGHT * 1.0)
        
        self.play(FadeIn(m_group), FadeIn(M_group))
        self.lecture[0].set_color("#4169E1")

        # === Animation for Lecture Line 2 ===
        # Move m toward M
        self.play(m_group.animate.shift(RIGHT * 2.0), run_time=1.5)
        self.lecture[1].set_color("#4169E1")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(collision_count, 'B3', scale_factor=0.8)
        self.play(FadeIn(collision_count))
        self.lecture[2].set_color("#FF4500")
