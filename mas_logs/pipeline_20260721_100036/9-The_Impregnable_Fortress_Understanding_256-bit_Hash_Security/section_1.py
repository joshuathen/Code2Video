from manim import *
import numpy as np
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
        # Data from storyboard
        title_text = "The Digital Treasure Chest"
        lecture_lines = [
            "- Meet Cipher, guarding a chest with 256 switches.",
            "- A cryptographic hash is like a digital fingerprint.",
            "- Every bit must match to unlock the treasure.",
            "- Guessing one bit wrong keeps the chest shut.",
            "- How hard is it to guess all 256?"
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_CIPHER = "#00FF00"
        COLOR_CHEST = "#FFD700"
        COLOR_BIT = "#FFFFFF"
        COLOR_LOCKED = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Meet Cipher, guarding a chest with 256 switches.
        self.lecture[0].set_color(YELLOW)
        
        # Cipher the Dragon: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dragon.svg]
        dragon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dragon.svg", color=COLOR_CIPHER, fill_opacity=0.8)
        dragon_label = Text("Cipher", font_size=16, color=COLOR_CIPHER).next_to(dragon, DOWN, buff=0.1)
        cipher_group = VGroup(dragon, dragon_label)
        # Issue 33: place at E2
        self.place_at_grid(cipher_group, 'E2', scale_factor=0.6)
        
        # Treasure Chest
        chest_box = Rectangle(width=1.0, height=0.7, color=COLOR_CHEST, fill_opacity=0.5)
        chest_lock = Dot(color=COLOR_CHEST).move_to(chest_box.get_center())
        chest_group = VGroup(chest_box, chest_lock)
        # Issue 33: place at E4
        self.place_at_grid(chest_group, 'E4', scale_factor=0.8)
        
        self.play(Create(cipher_group), Create(chest_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A cryptographic hash is like a digital fingerprint.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Fingerprint: Concentric Arcs
        fingerprint = VGroup(*[Arc(radius=0.1*i, start_angle=0, angle=PI*1.4, color=WHITE) for i in range(1, 5)])
        # Issue 31: place at B1
        self.place_at_grid(fingerprint, 'B1', scale_factor=0.6)
        self.play(Create(fingerprint))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Every bit must match to unlock the treasure.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # 256 slots: 16x16 grid of small squares
        slots = VGroup()
        for i in range(16):
            for j in range(16):
                slot = Square(side_length=0.05, color=WHITE, stroke_width=1)
                slots.add(slot)
        
        slots.arrange_in_grid(rows=16, cols=16, buff=0.03)
        # Issue 32: place_in_area 'A2', 'C6', scale 0.8
        self.place_in_area(slots, 'A2', 'C6', scale_factor=0.8)
        
        self.play(FadeIn(slots), FadeOut(fingerprint))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Guessing one bit wrong keeps the chest shut.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Random bits flashing to represent guessing
        flash_tracker = ValueTracker(0)
        
        def flash_slots_updater(mobject):
            # Use tracker value to vary the seed
            seed_val = int(flash_tracker.get_value() * 100)
            rng = random.Random(seed_val)
            for _ in range(40):
                idx = rng.randint(0, 255)
                # Toggle fill opacity
                mobject[idx].set_fill(WHITE, opacity=rng.random())

        slots.add_updater(flash_slots_updater)
        self.play(flash_tracker.animate.set_value(10), run_time=2)
        slots.remove_updater(flash_slots_updater)
        
        # Stop and show one failure
        for s in slots:
            s.set_fill(WHITE, opacity=0.8)
            
        bad_bit_idx = 100 # Arbitrary bit to fail
        self.play(
            slots[bad_bit_idx].animate.set_color(COLOR_LOCKED).set_fill(COLOR_LOCKED, opacity=1.0),
            run_time=0.5
        )
        
        # 'Locked' icon/label
        locked_text = Text("LOCKED", font_size=24, color=COLOR_LOCKED, weight=BOLD)
        # Issue 33: place at E5
        self.place_at_grid(locked_text, 'E5', scale_factor=1.0)
        self.play(Write(locked_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # How hard is it to guess all 256?
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Dragon breathes a small green flame
        flame_particles = VGroup(*[
            Triangle(color=COLOR_CIPHER, fill_opacity=0.8).scale(0.05) 
            for _ in range(8)
        ])
        
        # Initial positions at dragon's snout
        snout_pos = dragon.get_right()
        for p in flame_particles:
            p.move_to(snout_pos)
            
        def flame_updater(mobject, dt):
            for i, p in enumerate(mobject):
                # Particles move right at slightly different speeds
                speed = 2.5 + (i % 3) * 0.4
                p.shift(RIGHT * speed * dt)
                # Reset if they hit the chest area
                if p.get_x() > chest_group.get_x():
                    p.move_to(snout_pos)

        self.add(flame_particles)
        flame_particles.add_updater(flame_updater)
        self.wait(3)
        flame_particles.remove_updater(flame_updater)
        self.play(FadeOut(flame_particles))
        
        self.wait(2)
