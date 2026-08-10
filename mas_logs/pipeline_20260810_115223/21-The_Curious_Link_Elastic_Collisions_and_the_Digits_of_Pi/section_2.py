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
        self.setup_layout("The Collisional Setup", [
            "We track collisions between two blocks and a wall.",
            "Mass ratio changes the collision count.",
            "The hamster hits the sumo wrestler."
        ])
        
        # Elements
        wall = Line(UP, DOWN, color=GREY).scale(1.5)
        self.place_at_grid(wall, 'C2', scale_factor=0.9)
        wall_label = Text("Wall", color="#FFFFE0", font_size=20)
        wall_label.next_to(wall, LEFT)
        
        # Asset loading
        m_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hamster.svg")
        M_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sumo.svg")
        
        m_label = Text("m", color="#FFFFE0", font_size=20)
        M_label = Text("M", color="#FFFFE0", font_size=20)
        
        m_group = VGroup(m_block, m_label).arrange(UP, buff=0.1)
        M_group = VGroup(M_block, M_label).arrange(UP, buff=0.1)
        
        self.place_in_area(m_group, 'C4', 'C4', scale_factor=0.6)
        self.place_in_area(M_group, 'C5', 'D6', scale_factor=0.8)
        
        collision_count = 0
        counter_text = Text(f"Collisions: {collision_count}", color="#90EE90", font_size=24)
        self.place_at_grid(counter_text, 'B5', scale_factor=1.0)
        
        self.add(wall, wall_label, m_group, M_group, counter_text)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFE0")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#90EE90")
        for _ in range(3):
            self.play(m_group.animate.move_to(self.grid['C3']), run_time=0.5)
            self.play(m_group.animate.move_to(self.grid['C4']), run_time=0.5)
            collision_count += 1
            new_text = Text(f"Collisions: {collision_count}", color="#90EE90", font_size=24)
            self.place_at_grid(new_text, 'B5')
            self.remove(counter_text)
            counter_text = new_text
            self.add(counter_text)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFB6C1")
        self.play(M_group.animate.scale(1.5), run_time=1)
        for _ in range(5):
            self.play(m_group.animate.move_to(self.grid['C3']), run_time=0.2)
            self.play(m_group.animate.move_to(self.grid['C4']), run_time=0.2)
            collision_count += 1
            new_text = Text(f"Collisions: {collision_count}", color="#90EE90", font_size=24)
            self.place_at_grid(new_text, 'B5')
            self.remove(counter_text)
            counter_text = new_text
            self.add(counter_text)
        self.wait(1)
