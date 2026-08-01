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
        # Setup layout with updated lecture script
        self.setup_layout(
            "Prerequisite: The One-Way Function (Hashing)", 
            [
                'Every day, your phone generates a secret Daily Seed.', 
                'A cryptographic hash function transforms seeds into random IDs.', 
                'These IDs are one-way and protect your original seed.'
            ]
        )
        
        # Define Colors
        COLOR_SEED = "#5DADE2"
        COLOR_HASH = "#F4D336"
        COLOR_EPHID = "#58D68D"
        COLOR_WALL = "#EC7063"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_SEED))
        
        # Create Daily Seed
        seed_rect = Rectangle(width=1.5, height=0.8, color=COLOR_SEED, fill_opacity=0.3)
        seed_text = Text("Daily Seed", font_size=18, color=COLOR_SEED)
        seed_group = VGroup(seed_rect, seed_text)
        # Resolved Issue 35: Move to C2 and scale to 0.8
        self.place_at_grid(seed_group, "C2", scale_factor=0.8)
        
        # Create Hash Box
        hash_box = Rectangle(width=2, height=2, color=COLOR_HASH, fill_opacity=0.2)
        hash_label = Text("Hash\nFunction", font_size=20, color=COLOR_HASH)
        hash_group = VGroup(hash_box, hash_label)
        # Resolved Issue 36: Move to B4-D5 and scale to 0.9
        self.place_in_area(hash_group, "B4", "D5", scale_factor=0.9)
        
        self.play(FadeIn(seed_group))
        self.wait(1)
        
        # Move Seed into Hash Box
        self.play(seed_group.animate.move_to(hash_group.get_center()), run_time=1.5)
        self.play(FadeOut(seed_group), FadeIn(hash_group))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(COLOR_HASH))
        
        # Create Ephemeral IDs
        ephid_positions = ["B6", "C6", "D6"]
        ephids = VGroup()
        for i, pos in enumerate(ephid_positions):
            e_text = Text(f"EphID_{i+1}", font_size=18, color=COLOR_EPHID)
            e_box = RoundedRectangle(corner_radius=0.1, width=1.2, height=0.5, color=COLOR_EPHID)
            e_group = VGroup(e_box, e_text)
            self.place_at_grid(e_group, pos)
            ephids.add(e_group)

        # Animate EphIDs emerging from Hash Box
        spawn_animations = []
        for ephid in ephids:
            # Start from center of hash box
            target_pos = ephid.get_center().copy()
            ephid.move_to(hash_group.get_center())
            spawn_animations.append(ephid.animate.move_to(target_pos))
        
        self.play(AnimationGroup(*spawn_animations, lag_ratio=0.3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_EPHID))
        
        # Create One-Way Wall
        # Positioned at column 3 to sit between Seed (Col 2) and Hash/IDs (Col 4-6)
        wall = Line(self.grid["A3"], self.grid["F3"], color=COLOR_WALL, stroke_width=8)
        wall_label = Text("One-Way Wall", font_size=16, color=COLOR_WALL).next_to(wall, LEFT, buff=0.1)
        
        self.play(Create(wall), FadeIn(wall_label))
        
        # Create Back Arrow trying to go from IDs to Seed
        start_point = ephids[1].get_left()
        end_point = self.grid["C2"]
        back_arrow = Arrow(start_point, end_point, color=COLOR_WALL, buff=0)
        
        # Intersection with wall is at column 3
        collision_point = [self.grid["C3"][0], self.grid["C3"][1], 0]
        
        self.play(back_arrow.animate.put_start_and_end_on(start_point, collision_point), run_time=1)
        
        # Visual feedback for hitting the wall
        flash = Flash(collision_point, color=COLOR_WALL, flash_radius=0.3)
        self.play(flash)
        
        # Rebound slightly
        self.play(back_arrow.animate.shift(RIGHT * 0.2), run_time=0.5)
        
        self.wait(2)
